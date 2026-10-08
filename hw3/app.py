"""Interactive ReAct agent with Python, web search, and arXiv tools.

Requires the OPENAI_API_KEY environment variable to be set.
"""

import os
import sys

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"

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


def get_api_key():
    """Return the OpenAI API key from the OPENAI_API_KEY environment variable.

    Exits the program with an error message if the variable is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set the OPENAI_API_KEY environment variable first.")
    return api_key


def build_tools():
    """Create the agent's tools: a Python REPL, DuckDuckGo search, and arXiv search."""
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    return [PythonREPLTool(), DuckDuckGoSearchRun(), arxiv]


def build_agent(api_key):
    """Build an AgentExecutor that runs a ReAct agent over the tools from build_tools()."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=api_key)
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
    executor = build_agent(get_api_key())
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
