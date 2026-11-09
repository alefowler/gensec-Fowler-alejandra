"""
File Integrity Monitor (FIM).

A simple command-line tool that detects changes to files in a directory.

Usage:
    # Create a baseline of all file hashes in a directory
    python file_integrity_monitor.py /path/to/dir --mode baseline

    # Later, compare the directory against the saved baseline
    python file_integrity_monitor.py /path/to/dir --mode check

    # Use a custom baseline file location
    python file_integrity_monitor.py /path/to/dir --mode check --baseline my_baseline.json

In "baseline" mode, the tool walks the directory recursively, computes a
SHA-256 hash for every file, and saves the results to a JSON file.

In "check" mode, the tool recomputes the hashes and compares them with the
saved baseline, reporting files that were MODIFIED, ADDED, or DELETED.
The process exits with status 0 if nothing changed and 1 if changes were
found, so it can be used from scripts or cron jobs.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

DEFAULT_BASELINE_FILE = "baseline.json"
CHUNK_SIZE = 65536  # Read files in 64 KB chunks so large files don't fill memory.


def hash_file(file_path):
    """
    Compute the SHA-256 hash of a single file.

    The file is read in chunks so that very large files can be hashed
    without loading them entirely into memory.

    Args:
        file_path (str): Path to the file to hash.

    Returns:
        str: The hexadecimal SHA-256 digest of the file's contents.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_directory(directory, exclude=None):
    """
    Recursively hash every regular file in a directory.

    Paths in the result are stored relative to ``directory`` (with forward
    slashes) so the baseline stays valid even if the directory is checked
    from a different working directory. Symbolic links are skipped, and
    files that cannot be read are reported as warnings and skipped.

    Args:
        directory (str): The root directory to scan.
        exclude (str | None): Absolute path of a file to leave out of the
            scan (used so the baseline file is not hashed if it is stored
            inside the monitored directory).

    Returns:
        dict[str, str]: A mapping of relative file path -> SHA-256 hash.
    """
    hashes = {}
    exclude = os.path.abspath(exclude) if exclude else None

    for root, dirs, files in os.walk(directory):
        dirs.sort()  # Deterministic traversal order.
        for name in sorted(files):
            full_path = os.path.join(root, name)
            if os.path.islink(full_path) or not os.path.isfile(full_path):
                continue
            if exclude and os.path.abspath(full_path) == exclude:
                continue
            rel_path = os.path.relpath(full_path, directory).replace(os.sep, "/")
            try:
                hashes[rel_path] = hash_file(full_path)
            except OSError as e:
                print(f"[WARNING] Could not read {rel_path}: {e}", file=sys.stderr)
    return hashes


def save_baseline(directory, baseline_file):
    """
    Scan a directory and save its file hashes to a JSON baseline file.

    Args:
        directory (str): The directory to scan.
        baseline_file (str): Path of the JSON file to write.

    Returns:
        dict[str, str]: The file hashes that were saved.
    """
    hashes = scan_directory(directory, exclude=baseline_file)
    data = {
        "directory": os.path.abspath(directory),
        "created": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sha256",
        "files": hashes,
    }
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return hashes


def load_baseline(baseline_file):
    """
    Load a previously saved baseline from a JSON file.

    Args:
        baseline_file (str): Path of the JSON baseline file.

    Returns:
        dict: The baseline data, including a ``"files"`` mapping of
        relative path -> SHA-256 hash.

    Raises:
        FileNotFoundError: If the baseline file does not exist.
        ValueError: If the file is not a valid baseline.
    """
    with open(baseline_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise ValueError(f"{baseline_file} is not a valid baseline file")
    return data


def compare_hashes(baseline, current):
    """
    Compare baseline hashes against current hashes.

    Args:
        baseline (dict[str, str]): Path -> hash mapping from the baseline.
        current (dict[str, str]): Path -> hash mapping from the current scan.

    Returns:
        dict[str, list[str]]: A dictionary with sorted lists under the keys
        ``"modified"``, ``"added"``, and ``"deleted"``.
    """
    baseline_paths = set(baseline)
    current_paths = set(current)
    return {
        "modified": sorted(
            p for p in baseline_paths & current_paths if baseline[p] != current[p]
        ),
        "added": sorted(current_paths - baseline_paths),
        "deleted": sorted(baseline_paths - current_paths),
    }


def check_integrity(directory, baseline_file):
    """
    Re-scan a directory and compare it against the saved baseline.

    Args:
        directory (str): The directory to scan.
        baseline_file (str): Path of the JSON baseline file to compare with.

    Returns:
        dict[str, list[str]]: The changes found (see ``compare_hashes``).
    """
    baseline = load_baseline(baseline_file)
    current = scan_directory(directory, exclude=baseline_file)
    return compare_hashes(baseline["files"], current)


def print_report(changes):
    """
    Print a human-readable report of integrity changes.

    Args:
        changes (dict[str, list[str]]): Output of ``compare_hashes``.
    """
    total = sum(len(paths) for paths in changes.values())
    if total == 0:
        print("[OK] No changes detected. All files match the baseline.")
        return

    print(f"[ALERT] {total} change(s) detected:")
    for label in ("modified", "added", "deleted"):
        for path in changes[label]:
            print(f"  {label.upper():<9} {path}")


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Args:
        argv (list[str] | None): Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simple File Integrity Monitor using SHA-256 hashes."
    )
    parser.add_argument("directory", help="Directory to monitor")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["baseline", "check"],
        required=True,
        help="'baseline' to create a new baseline, 'check' to compare against it",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        default=DEFAULT_BASELINE_FILE,
        help=f"Path to the baseline JSON file (default: {DEFAULT_BASELINE_FILE})",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """
    Entry point for the command-line interface.

    Args:
        argv (list[str] | None): Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Exit status — 0 if successful with no changes, 1 if changes
        were detected, 2 on errors.
    """
    args = parse_args(argv)

    if not os.path.isdir(args.directory):
        print(f"[ERROR] Not a directory: {args.directory}", file=sys.stderr)
        return 2

    if args.mode == "baseline":
        hashes = save_baseline(args.directory, args.baseline)
        print(f"[OK] Baseline saved to {args.baseline} ({len(hashes)} files hashed).")
        return 0

    try:
        changes = check_integrity(args.directory, args.baseline)
    except FileNotFoundError:
        print(
            f"[ERROR] Baseline file not found: {args.baseline}. "
            "Run with --mode baseline first.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not load baseline: {e}", file=sys.stderr)
        return 2

    print_report(changes)
    return 1 if any(changes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
