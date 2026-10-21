"""
MCP agent coordinating two servers: a custom text-tools server
(Caesar cipher + text stats) and the official Fetch MCP server
for retrieving web page content.

Requires GOOGLE_API_KEY and GOOGLE_MODEL environment variables.
"""

import os
import asyncio
from contextlib import AsyncExitStack
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL"))

SYSTEM_PROMPT = (
    "You are an assistant with access to two tool servers: one with "
    "a Caesar cipher encoder/decoder and a text statistics tool, and "
    "one that can fetch the text content of a webpage. Use the tools "
    "together when needed to answer multi-step requests."
)


async def run_agent():
    """Connect to both MCP servers, merge their tools, and run an interactive loop."""
    text_tools_params = StdioServerParameters(command="uv", args=["run", "mcp-server.py"])
    fetch_params = StdioServerParameters(command="uvx", args=["mcp-server-fetch"])

    async with AsyncExitStack() as stack:
        text_read, text_write = await stack.enter_async_context(stdio_client(text_tools_params))
        text_session = await stack.enter_async_context(ClientSession(text_read, text_write))
        await text_session.initialize()
        text_tools = await load_mcp_tools(text_session)

        fetch_read, fetch_write = await stack.enter_async_context(stdio_client(fetch_params))
        fetch_session = await stack.enter_async_context(ClientSession(fetch_read, fetch_write))
        await fetch_session.initialize()
        fetch_tools = await load_mcp_tools(fetch_session)

        all_tools = text_tools + fetch_tools
        agent = create_agent(model=llm, tools=all_tools, system_prompt=SYSTEM_PROMPT)
        history = []

        print("Welcome to the Text Toolkit Agent. Type a request, or an empty line to quit.")

        while True:
            line = input("llm>> ")
            if not line:
                break
            try:
                history.append(HumanMessage(content=line))
                result = await agent.ainvoke({"messages": history})
                if "messages" in result and result["messages"]:
                    response_content = StrOutputParser().invoke(result["messages"][-1])
                    history.append(AIMessage(content=response_content))
                    print(response_content)
                else:
                    print("No response from agent")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_agent())
