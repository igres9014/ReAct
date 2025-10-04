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
import importlib
import json
import datetime
import time
import logging
from typing import Any
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

# reduce noisy debug/info from langgraph/langchain adapters
logging.getLogger("langgraph").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.WARNING)

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception as e:
    raise ImportError("langchain_mcp_adapters is required; install it (pip install langchain-mcp-adapters) or ensure it's on PYTHONPATH") from e

# explicit OpenAI SDK import for embeddings / client usage
try:
    from openai import OpenAI
except Exception:
    # avoid hard crash here; main will still error if OPENAI API is needed
    OpenAI = None

# LangChain / LangGraph bits
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

try:
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    _LANGSMITH_CLIENT = None
    if LANGSMITH_API_KEY:
        _ls = importlib.import_module("langsmith")
        ClientCls = getattr(_ls, "Client", None)
        if ClientCls:
            try:
                _LANGSMITH_CLIENT = ClientCls(api_key=LANGSMITH_API_KEY)
            except Exception:
                _LANGSMITH_CLIENT = None
except Exception:
    _LANGSMITH_CLIENT = None

def _trace_to_langsmith(prompt: str, response: str, raw: Any = None):
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt": prompt,
        "response": response,
    }
    if raw is not None:
        try:
            payload["raw"] = json.loads(json.dumps(raw, default=str))
        except Exception:
            payload["raw"] = str(raw)
    # try langsmith client
    try:
        if _LANGSMITH_CLIENT and hasattr(_LANGSMITH_CLIENT, "create_run"):
            try:
                _LANGSMITH_CLIENT.create_run(payload)
                return
            except Exception:
                pass
    except Exception:
        pass
    # fallback: write local trace
    try:
        os.makedirs("langsmith_traces", exist_ok=True)
        fname = f"langsmith_traces/run_{int(time.time())}.json"
        with open(fname, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

# LangChain callback handler (best-effort, optional)
_lc_cb = None
try:
    from langchain.callbacks.base import BaseCallbackHandler
    class _LangChainTraceHandler(BaseCallbackHandler):
        def __init__(self):
            self._prompts = []

        def on_llm_start(self, serialized, prompts, **kwargs):
            try:
                self._prompts = list(prompts or [])
            except Exception:
                self._prompts = []

        def on_llm_end(self, response, **kwargs):
            try:
                text = ""
                gens = getattr(response, "generations", None)
                if gens:
                    parts = []
                    for g in gens:
                        if isinstance(g, list):
                            parts.append(" | ".join([getattr(x, "text", str(x)) for x in g]))
                        else:
                            parts.append(getattr(g, "text", str(g)))
                    text = "\n".join(parts)
                else:
                    text = str(response)
                prompt = self._prompts[0] if self._prompts else ""
                _trace_to_langsmith(prompt, text, raw=response)
            except Exception:
                pass

    _lc_cb = _LangChainTraceHandler()
except Exception:
    _lc_cb = None

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
    # attach LangChain callback handler if available to trace LLM calls to Langsmith
    if _lc_cb:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[_lc_cb])
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3) Create a ReAct agent graph with the MCP tools
    #    (tools are already LangChain-compatible from the adapter)
    # Turn off internal debug to avoid printing internal updates/values objects.
    agent = create_react_agent(llm, milvus_tools, debug=False)
    return agent, mcp_client, milvus_tools


async def _call_vector_search_with_variants(search_tool, collection: str, vector: list, top_k: int = 5):
    """
    Best-effort: try several common payload shapes for MCP search tools until one succeeds.
    Returns tool result or raises the last exception.
    """
    variants = [
        {"collection": collection, "vector": vector, "top_k": top_k},
        {"collection": collection, "query_vector": vector, "top_k": top_k},
        {"collection": collection, "embeddings": [vector], "n_results": top_k},
        {"collection": collection, "vectors": [vector], "top_k": top_k},
        {"collection": collection, "vector": vector, "k": top_k},
        {"collection": collection, "query": {"vector": vector}, "top_k": top_k},
    ]
    last_exc = None
    for payload in variants:
        try:
            res = await search_tool.ainvoke(payload)
            return {"payload": payload, "result": res}
        except Exception as e:
            last_exc = e
            # continue trying other shapes
            continue
    raise last_exc or RuntimeError("No payload variant succeeded for search tool")


async def demo_run(agent, milvus_tools):
    """
    Run a couple of sample prompts to exercise Milvus MCP tools through ReAct.
    Uses structured tool calls for vector search to avoid schema/field-name bugs.
    """
    # Example 1: list collections (via agent)
    print("\n--- Example: list collections ---")
    result = await agent.ainvoke({"messages": [HumanMessage(content="List my Milvus collections.")]})
    out_text = result["messages"][-1].content
    print(out_text)
    try:
        _trace_to_langsmith("List my Milvus collections.", out_text, raw=result)
    except Exception:
        pass

    # Example 2: vector/text search (human-in-the-loop), try text-first then vector variants
    print("\n--- Example: vector/text search (human-in-the-loop) ---")
    try:
        collection = input("Collection name to search (e.g. PROVA_33): ").strip() or "your_collection"
        query_text = input('Query text (e.g. "what is chroma?"): ').strip()
        if not query_text:
            print("No query text provided; aborting search example.")
            return

        # find a search-like tool
        search_tool = None
        for t in milvus_tools:
            name = getattr(t, "name", "") or str(t)
            if "search" in name.lower() or "query" in name.lower() or "text" in name.lower():
                search_tool = t
                break
        if not search_tool:
            print("No search/query tool found among MCP tools. Skipping search example.")
            return

        # First: try text-search payload shapes (many MCP tools expose text search)
        tried = []
        text_variants = [
            {"collection_name": collection, "query_text": query_text, "top_k": 5},
            {"collection": collection, "query_text": query_text, "top_k": 5},
            {"collection_name": collection, "query": query_text, "k": 5},
            {"collection": collection, "query": query_text, "n_results": 5},
        ]
        for payload in text_variants:
            tried.append(payload)
            try:
                resp = await search_tool.ainvoke(payload)
                print("Text-search succeeded with payload:", payload)
                print("Result:", resp)
                try:
                    _trace_to_langsmith(query_text, json.dumps(resp, default=str), raw=resp)
                except Exception:
                    pass
                return
            except Exception as e:
                # attempt auto-repair using pydantic validation errors (best-effort)
                repaired = False
                try:
                    errs = e.errors() if hasattr(e, "errors") else None
                except Exception:
                    errs = None
                missing_keys = []
                if errs:
                    for er in errs:
                        loc = er.get("loc") or er.get("input") or er.get("type")
                        if isinstance(loc, (list, tuple)) and loc:
                            missing_keys.append(loc[0])
                        elif isinstance(loc, str):
                            missing_keys.append(loc)
                else:
                    # fallback: parse message heuristically
                    msg = str(e)
                    import re
                    missing_keys += re.findall(r"([a-zA-Z_]+)\\s+\\n\\s+Missing required argument", msg)

                if missing_keys:
                    # build a minimal payload satisfying missing keys
                    auto = {}
                    for k in set(missing_keys):
                        lk = k.lower()
                        if "collection" in lk:
                            auto[k] = collection
                        elif "query_text" in lk or ("query" in lk and "text" in lk):
                            auto[k] = query_text
                        elif "query" in lk and "vector" in lk:
                            auto[k] = {"vector": vector} if 'vector' in locals() else query_text
                        elif "k" in lk or "top" in lk or "n_results" in lk:
                            auto[k] = 5
                        elif "vector" in lk or "emb" in lk:
                            auto[k] = vector if 'vector' in locals() else None
                    # merge with tried payloads to include other params
                    for p in reversed(tried):
                        for kk, vv in p.items():
                            if kk not in auto:
                                auto[kk] = vv
                    try:
                        resp = await search_tool.ainvoke(auto)
                        print("Text-search succeeded with repaired payload:", auto)
                        print("Result:", resp)
                        try:
                            _trace_to_langsmith(query_text, json.dumps(resp, default=str), raw=resp)
                        except Exception:
                            pass
                        return
                    except Exception:
                        repaired = False
                # otherwise continue trying other variants
                continue

        # If text variants failed, fall back to vector search variants
        if OpenAI is None:
            print("OpenAI SDK not available, cannot compute embeddings for vector search. Install openai package or abort.")
            return
        oai = OpenAI()
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        emb_resp = oai.embeddings.create(model=embed_model, input=[query_text])
        vector = emb_resp.data[0].embedding

        try:
            search_out = await _call_vector_search_with_variants(search_tool, collection, vector, top_k=5)
            print("Vector-search succeeded with payload:", search_out["payload"])
            print("Search result:", search_out["result"])
            try:
                _trace_to_langsmith(query_text, json.dumps(search_out["result"], default=str), raw=search_out)
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            print("Vector search failed:", msg)
            if "fieldName(sparse)" in msg or "fieldName" in msg:
                print("\nHint: the target collection may not have the expected vector field. Recreate the collection with a vector field (common names: 'embeddings', 'vector', 'sparse').")
            else:
                print("\nSearch tool returned an error. Inspect MCP server logs for details.")
    except KeyboardInterrupt:
        print("\nInterrupted by user, aborting demo.")
    except Exception as e:
        print("Demo search failed:", e)

async def main():
    # CLI: optional CSV ingestion args
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--csv", help="Path to CSV file to ingest")
    ap.add_argument("--id-column", help="CSV column to use as id")
    ap.add_argument("--text-column", help="CSV column to use as text/content")
    ap.add_argument("--metadata-columns", help="Comma-separated CSV columns to include as metadata (default: all other columns)")
    ap.add_argument("--collection", help="Target collection name for ingest (default: your MCP/milvus collection)", default="your_collection")
    parsed, _ = ap.parse_known_args()

    agent, mcp_client, _tools = await build_agent()

    try:
        # If CSV ingestion requested, run it before demos
        if parsed.csv:
            meta_cols = [c.strip() for c in parsed.metadata_columns.split(",")] if parsed.metadata_columns else None
            try:
                # milvus tools are loaded inside build_agent -> _tools
                # Use the explicit OpenAI import if available; otherwise warn and skip local embeddings
                if OpenAI is None:
                    print("WARNING: OpenAI SDK not available; CSV ingest will call MCP tool without local embeddings.")
                    oai_client = None
                else:
                    oai_client = OpenAI()
                ingest_res = await maybe_ingest_csv_to_mcp(_tools, oai_client, parsed.csv, parsed.collection, parsed.id_column, parsed.text_column, meta_cols)
                print("CSV ingest result:", ingest_res)
            except Exception as e:
                print("CSV ingest failed:", e)

        # Option A: simple invoke
        await demo_run(agent, _tools)

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
        # Robust shutdown for MultiServerMCPClient: try common async/sync cleanup methods.
        async def _close_client(client):
            if client is None:
                return
            # try a list of likely teardown method names
            for name in ("aclose", "close", "shutdown", "stop", "disconnect", "terminate", "close_all"):
                fn = getattr(client, name, None)
                if not fn:
                    continue
                try:
                    if asyncio.iscoroutinefunction(fn):
                        await fn()
                    else:
                        res = fn()
                        if asyncio.iscoroutine(res):
                            await res
                    return
                except Exception:
                    # ignore and try next available method
                    continue

        await _close_client(mcp_client)


async def maybe_ingest_csv_to_mcp(tools, oai_client, csv_path, collection, id_column, text_column, metadata_columns):
    """
    Minimal stub for CSV ingest via MCP tools.
    Looks for an upsert-like tool and calls it with a best-effort payload.
    Replace/extend with full CSV-reading + embedding logic as needed.
    """
    # find an upsert-like tool
    insert_tool = None
    for t in tools:
        name = getattr(t, "name", "") or str(t)
        if any(k in name.lower() for k in ("insert", "upsert", "add", "ingest")):
            insert_tool = t
            break
    if not insert_tool:
        return {"status": "no_upsert_tool_found", "path": csv_path}

    # Best-effort: call tool with file path so the MCP server can handle ingestion,
    # or pass a minimal payload. Many MCP tools expect documents/ids => adapt as needed.
    payload = {"collection": collection, "csv_path": str(csv_path), "id_column": id_column, "text_column": text_column, "metadata_columns": metadata_columns}
    try:
        # tool.ainvoke is async in these adapters
        res = await insert_tool.ainvoke(payload)
        return {"status": "called_upsert_tool", "tool": getattr(insert_tool, "name", str(insert_tool)), "result": res}
    except Exception as e:
        return {"status": "tool_call_failed", "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
