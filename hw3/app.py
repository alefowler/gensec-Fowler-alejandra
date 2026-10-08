"""Interactive ReAct agent with Python, web search, and arXiv tools.

Requires the GOOGLE_API_KEY and GOOGLE_MODEL environment variables to be set.
"""

import os
import sys

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_google_genai import ChatGoogleGenerativeAI

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def get_env(name):
    """Return the value of the environment variable `name`.

    Exits the program with an error message if the variable is not set.
    """
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Error: set the {name} environment variable first.")
    return value


def build_tools():
    """Create the agent's tools: a Python REPL, DuckDuckGo search, and arXiv search."""
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    return [PythonREPLTool(), DuckDuckGoSearchRun(), arxiv]


def build_agent(model, api_key):
    """Build an AgentExecutor that runs a Gemini ReAct agent over the tools from build_tools()."""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=api_key)
    tools = build_tools()
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )


def main():
    """Read questions from the user in a loop and print the agent's answers."""
    executor = build_agent(get_env("GOOGLE_MODEL"), get_env("GOOGLE_API_KEY"))
    print("Ask a question (type 'exit' or 'quit' to stop).")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            result = executor.invoke({"input": question})
            print(f"\nAgent: {result['output']}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
