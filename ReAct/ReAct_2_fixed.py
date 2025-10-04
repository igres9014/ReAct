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

MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")


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
    # Prefer the schema observed in your error: collection_name + query_vector + limit
    variants = [
        {"collection_name": collection, "query_vector": vector, "limit": top_k},
        {"collection_name": collection, "vector": vector, "limit": top_k},
        {"collection_name": collection, "embeddings": [vector], "limit": top_k},
        {"collection_name": collection, "vectors": [vector], "limit": top_k},
        {"collection_name": collection, "query": {"vector": vector}, "limit": top_k},
        # Back-compat fallbacks for other possible tool impls
        {"collection_name": collection, "query_vector": vector, "k": top_k},
        {"collection_name": collection, "query_vector": vector, "top_k": top_k},
    ]
    last_exc = None
    for payload in variants:
        try:
            res = await search_tool.ainvoke(payload)
            return {"payload": payload, "result": res}
        except Exception as e:
            last_exc = e
            continue
    raise last_exc or RuntimeError("No payload variant succeeded for search tool")


async def ensure_hybrid_collection(milvus_tools, collection: str, dense_dim: int, sparse_dim: Optional[int] = None, drop: bool = False):
    """
    Best-effort: ensure a collection exists with both a dense vector field (embeddings)
    and a sparse vector field named 'sparse'. Attempts multiple MCP payload shapes for
    create_index/create_collection tools and for creating the sparse index (BM25/SPLADE).
    Returns a dict with the results of attempted operations.
    """
    res = {"collection": collection, "created": False, "index_created": False, "attempts": []}

    # find create-like tool
    create_tool = None
    for t in milvus_tools:
        n = getattr(t, "name", "") or str(t)
        if any(k in n.lower() for k in ("create_collection", "create", "milvus_create_collection", "collection.create")):
            create_tool = t
            break

    # find index-like tool
    index_tool = None
    for t in milvus_tools:
        n = getattr(t, "name", "") or str(t)
        if any(k in n.lower() for k in ("create_index", "createindex", "milvus_create_index", "index.create")):
            index_tool = t
            break

    # assemble candidate schemas/payloads
    sparse_dim_guess = sparse_dim or min(65536, max(1024, dense_dim * 16))
    create_variants = [
        # variant: explicit schema
        {
            "collection_name": collection,
            "schema": {
                "auto_id": False,
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True},
                    {"name": "embeddings", "type": "FLOAT_VECTOR", "params": {"dim": dense_dim}},
                    {"name": "sparse", "type": "SPARSE_FLOAT_VECTOR", "params": {"dim": sparse_dim_guess}},
                ],
            },
            "drop_if_exists": drop,
        },
        # variant: flatter payload
        {
            "collection_name": collection,
            "fields": [
                {"name": "id", "type": "INT64", "is_primary": True},
                {"name": "embeddings", "type": "FLOAT_VECTOR", "dim": dense_dim},
                {"name": "sparse", "type": "SPARSE_FLOAT_VECTOR", "dim": sparse_dim_guess},
            ],
            "drop": drop,
        },
        # variant: minimal params (some adapters expect this)
        {"collection": collection, "dim": dense_dim, "sparse_dim": sparse_dim_guess, "drop": drop},
    ]

    if create_tool:
        for payload in create_variants:
            try:
                res_call = await create_tool.ainvoke(payload)
                res["attempts"].append({"action": "create", "payload": payload, "result": res_call})
                res["created"] = True
                break
            except Exception as e:
                res["attempts"].append({"action": "create", "payload": payload, "error": str(e)})
    else:
        res["attempts"].append({"action": "create", "error": "no create tool found"})

    # attempt to create sparse/BM25 index if index tool exists and collection now present
    if index_tool and res["created"]:
        index_variants = [
            {"collection_name": collection, "field_name": "sparse", "index_type": "BM25", "params": {}},
            {"collection_name": collection, "field_name": "sparse", "index_type": "SPLADE", "params": {}},
            {"collection_name": collection, "field_name": "sparse", "index_type": "SPARSE_INDEX", "params": {}},
            # fallback: explicit name commonly used by some adapters
            {"collection": collection, "field": "sparse", "type": "BM25", "params": {}},
        ]
        for idx_payload in index_variants:
            try:
                idx_res = await index_tool.ainvoke(idx_payload)
                res["attempts"].append({"action": "create_index", "payload": idx_payload, "result": idx_res})
                res["index_created"] = True
                break
            except Exception as e:
                res["attempts"].append({"action": "create_index", "payload": idx_payload, "error": str(e)})
    else:
        if not index_tool:
            res["attempts"].append({"action": "create_index", "error": "no index tool found"})
    return res


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

        # First: try text-search payload shapes
        tried = []
        text_variants = [
            # Based on error message, the expected args are collection_name + query_text + limit
            {"collection_name": collection, "query_text": query_text, "limit": 5},
            # Some servers might accept these alternates:
            {"collection_name": collection, "query": query_text, "limit": 5},
            {"collection_name": collection, "query_text": query_text, "k": 5},
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
                        if "collection_name" in lk or ("collection" in lk and "name" in lk):
                            auto[k] = collection
                        elif "collection" in lk:
                            # prefer the canonical name
                            auto["collection_name"] = collection
                        elif "query_text" in lk:
                            auto[k] = query_text
                        elif lk == "query":
                            auto[k] = query_text
                        elif "limit" in lk:
                            auto[k] = 5
                        elif "k" in lk or "top" in lk or "n_results" in lk:
                            # normalize to limit; some servers alias it
                            auto["limit"] = 5
                        elif "vector" in lk or "emb" in lk:
                            # will be added in vector stage
                            pass
                    # ensure we use the expected keys
                    auto.setdefault("collection_name", collection)
                    auto.setdefault("query_text", query_text)
                    auto.setdefault("limit", 5)

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
        # ensure hybrid collection exists (best-effort)
        embed_model = os.getenv("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        dense_dim = MODEL_DIMS.get(embed_model, MODEL_DIMS[DEFAULT_EMBED_MODEL])
        try:
            print(f"[info] ensuring hybrid collection '{parsed.collection}' exists with dense dim {dense_dim} and sparse field 'sparse' ...")
            ensure_res = await ensure_hybrid_collection(_tools, parsed.collection, dense_dim, sparse_dim=None, drop=False)
            print("ensure_hybrid_collection result:", ensure_res)
        except Exception as e:
            print("ensure_hybrid_collection failed:", e)

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
    payload = {"collection_name": collection, "csv_path": str(csv_path), "id_column": id_column, "text_column": text_column, "metadata_columns": metadata_columns}
    try:
        # tool.ainvoke is async in these adapters
        res = await insert_tool.ainvoke(payload)
        return {"status": "called_upsert_tool", "tool": getattr(insert_tool, "name", str(insert_tool)), "result": res}
    except Exception as e:
        return {"status": "tool_call_failed", "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
