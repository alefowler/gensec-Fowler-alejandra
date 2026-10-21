"""
Custom MCP server implementing simple text-processing tools.

Tools:
- caesar_encode: Encode text using a Caesar cipher shift.
- caesar_decode: Decode text using a Caesar cipher shift.
- text_stats: Return word count, character count, and line count for text.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Custom Text Tools")


def _shift_char(ch: str, shift: int) -> str:
    """Shift a single alphabetic character by `shift` positions, wrapping around the alphabet. Non-alphabetic characters are returned unchanged."""
    if ch.isupper():
        return chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
    if ch.islower():
        return chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
    return ch


@mcp.tool("caesar_encode")
async def caesar_encode(text: str, shift: int) -> str:
    """
    Encode text using a Caesar cipher with the given shift.

    Args:
        text: The plaintext to encode.
        shift: Number of positions to shift each letter.

    Returns:
        The encoded ciphertext.
    """
    return "".join(_shift_char(c, shift) for c in text)


@mcp.tool("caesar_decode")
async def caesar_decode(text: str, shift: int) -> str:
    """
    Decode text that was encoded using a Caesar cipher with the given shift.

    Args:
        text: The ciphertext to decode.
        shift: Number of positions the text was originally shifted by.

    Returns:
        The decoded plaintext.
    """
    return "".join(_shift_char(c, -shift) for c in text)


@mcp.tool("text_stats")
async def text_stats(text: str) -> str:
    """
    Compute basic statistics for a block of text.

    Args:
        text: The text to analyze.

    Returns:
        A string summarizing word count, character count, and line count.
    """
    words = len(text.split())
    chars = len(text)
    lines = len(text.splitlines()) or 1
    return f"Words: {words}, Characters: {chars}, Lines: {lines}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        mcp.run(transport="http", host="0.0.0.0", port=8080)
    else:
        mcp.run(transport="stdio")
