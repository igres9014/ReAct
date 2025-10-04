# --- requirements (examples) ---
# pip install langgraph langchain-mcp-adapters "langchain>=0.2.16" langchain-openai
# (plus your OpenAI provider / other LLM provider libs)
#
# Start the Milvus MCP server first (pick one):
#   Stdio: uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530
#   SSE  : uv run src/mcp_server_milvus/server.py --sse --milvus-uri http://localhost:19530 --port 8000
#
# ENV:
#   export OPENAI_API_KEY=...



# Instructions to setup and run MILVUS:

# 1. Start MILVUS server. RUN from root:
# bash standalone_embed.sh start  
# manage docker: docker ps & docker stop XXXX


# 2. Start MILVUS MCP server. RUN from root:
# cd mcp-server-milvus && uv run src/mcp_server_milvus/server.py --milvus-uri http://localhost:19530


import os
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

# LangChain / LangGraph bits
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# --- Choose ONE connection config ---

# 1) Connect over stdio (spawns the Milvus MCP server process)
MILVUS_STDIO = {
    "milvus": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "--directory",
            "/home/sergi/mcp-server-milvus/src/mcp_server_milvus/",
            "run",
            "server.py",
            "--milvus-uri",
            "http://localhost:19530",
        ],
        # Optional: env vars for the spawned server
        # "env": {"MILVUS_API_KEY": "..."},
    }
}

# 2) Connect to an already-running SSE server
MILVUS_SSE = {
    "milvus": {
        "transport": "sse",
        "url": "http://localhost:8000/sse",
        # "headers": {"Authorization": "Bearer <TOKEN>"},
    }
}

CONNECTIONS = MILVUS_STDIO  # or MILVUS_SSE


async def build_agent():
    """
    Create a LangGraph ReAct agent wired up with MCP (Milvus) tools.
    Returns: (agent, mcp_client, tools)
    """
    # 1) Spin up the MCP multi-server client and load Milvus tools
    mcp_client = MultiServerMCPClient(CONNECTIONS)

    # You can load a subset by server_name or load all servers if needed
    milvus_tools = await mcp_client.get_tools(server_name="milvus")

    # 2) Pick an LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3) Create a ReAct agent graph with the MCP tools
    #    (tools are already LangChain-compatible from the adapter)
    agent = create_react_agent(llm, milvus_tools, debug=True)
    return agent, mcp_client, milvus_tools


async def demo_run(agent):
    """
    Run a couple of sample prompts to exercise Milvus MCP tools through ReAct.
    Adjust to match your collection and schema.
    """
    # Example 1: list collections
    print("\n--- Example: list collections ---")
    result = await agent.ainvoke({"messages": [HumanMessage(content="List my Milvus collections.")]})
    print(result["messages"][-1].content)

    # Example 2: vector search (adjust collection/vector dim/field names)
    print("\n--- Example: vector search ---")
    user_query = (
        "In the 'your_collection' collection, run a vector search with this 128-dim vector of zeros; "
        "return top 5 with ids and scores."
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_query)]})
    print(result["messages"][-1].content)


async def main():
    agent, mcp_client, _tools = await build_agent()

    try:
        # Option A: simple invoke
        await demo_run(agent)

        # Option B: stream tokens / tool calls live (nice for UIs or tracing)
        # print("\n--- Streaming example ---")
        # async for event in agent.astream_events(
        #     {"messages": [HumanMessage(content="What collections exist in Milvus?")]},
        #     version="v2",
        # ):
        #     kind = event["event"]
        #     if kind == "on_tool_start":
        #         print(f"[tool start] {event.get('name')}")
        #     elif kind == "on_tool_end":
        #         print(f"[tool end]   {event.get('name')}")
        #     elif kind == "on_chat_model_stream":
        #         print(event["data"]["chunk"].content, end="", flush=True)
        # print()

    finally:
        # Always close MCP client to tear down subprocesses / sessions
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
