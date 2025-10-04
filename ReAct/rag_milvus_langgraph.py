#!/usr/bin/env python3
"""
rag_milvus_langgraph.py  (v1)

End-to-end, human-in-the-loop RAG agent that:
- Connects to Milvus and searches collections like those created by milvus_query_interactive.py
- Uses LangGraph's prebuilt ReAct agent
- Loads additional MCP tools via MultiServerMCPClient (optional)
- Employs an LLM for query refinement + summarization
- Keeps a human in the loop to confirm/adjust the query and search parameters

Requirements (tested 2025-09):
  pip install --upgrade pymilvus pandas langchain langgraph langchain-openai langchain-core pydantic openai
  # For MCP integration:
  pip install --upgrade langchain-mcp-adapters  # exposes MultiServerMCPClient
  # If you also want the official MCP SDKs / examples:
  # pip install modelcontextprotocol

Environment:
  OPENAI_API_KEY         - for LLM + embeddings
  MILVUS_HOST (optional) - default 127.0.0.1
  MILVUS_PORT (optional) - default 19530

Usage:
  python rag_milvus_langgraph.py \
     --collection <your_collection> \
     --host 127.0.0.1 \
     --port 19530 \
     --chat-model gpt-4o-mini \
     --embed-model text-embedding-3-small \
     --mcp-config ./mcp_config.json

mcp_config.json example:
{
  "math": {
    "transport": "stdio",
    "command": "python",
    "args": ["./examples/math_server.py"]
  },
  "weather": {
    "transport": "streamable_http",
    "url": "http://localhost:8000/mcp/"
  }
}
(The MCP config is optional. If provided, the agent can also use those tools.)

Workflow:
  1) You type a natural language need.
  2) LLM proposes 2-3 precise search queries.
  3) You pick one or edit it.
  4) The agent calls a local Milvus tool (semantic/keyword/hybrid) and returns Top-K + summary.
  5) Loop for more queries.
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from collections import Counter

# Milvus
from pymilvus import connections, Collection

# LangGraph + LangChain
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# MCP client (optional)
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
    HAVE_MCP = True
except Exception:
    HAVE_MCP = False


# ------------------------------- Helpers (BM25, tokenization) -------------------------------

def tokenize(text: str) -> List[str]:
    return [tok for tok in ''.join(
        ch.lower() if ch.isalnum() else ' ' for ch in (text or "")
    ).split() if tok]


def bm25_prepare(docs: List[str], k: float = 1.2, b: float = 0.75):
    tokenized = [tokenize(t) for t in docs]
    N = len(tokenized) or 1
    avgdl = (sum(len(toks) for toks in tokenized) / N) if N else 0.0
    dfreq = Counter()
    for toks in tokenized:
        dfreq.update(set(toks))
    idf = {t: math.log((N - dfreq[t] + 0.5) / (dfreq[t] + 0.5) + 1) for t in dfreq}
    return tokenized, idf, avgdl, k, b


def bm25_score(query: str, tokenized_docs: List[List[str]], idf: Dict[str, float],
               avgdl: float, k: float, b: float) -> List[float]:
    q_terms = tokenize(query)
    scores = []
    for toks in tokenized_docs:
        tf = Counter(toks)
        dl = len(toks) or 1
        s = 0.0
        for term in q_terms:
            if term not in idf:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += idf[term] * ((f * (k + 1)) / (f + k * (1 - b + b * dl / (avgdl or 1.0))))
        scores.append(float(s))
    return scores


def minmax_norm(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    vmin, vmax = min(vals), max(vals)
    if vmax <= vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


# -------------------------------- Milvus wrappers --------------------------------

def pick_text_field(collection: Collection) -> str:
    """
    Prefer 'text' if present; else a VARCHAR-like field other than dense/sparse/id.
    """
    schema = collection.schema
    for f in schema.fields:
        if getattr(f, "name", None) == "text":
            return "text"
    def is_varchar(field) -> bool:
        dtype_name = getattr(field, "dtype", None)
        dtype_name = getattr(dtype_name, "name", str(dtype_name))
        return str(dtype_name).upper() in ("VARCHAR", "STRING")
    for f in schema.fields:
        if is_varchar(f) and f.name not in ("dense", "sparse", "id"):
            return f.name
    return "text"


def fetch_by_ids(collection: Collection, ids: List[int], fields: List[str]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not ids:
        return out
    batch = 1000
    id_col = "id"
    want = list(dict.fromkeys([id_col] + fields))
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        expr = f"{id_col} in {chunk}"
        rows = collection.query(expr=expr, output_fields=want)
        for r in rows:
            out[int(r[id_col])] = r
    return out


def milvus_semantic_search(collection: Collection, vector: List[float], top_k: int,
                           expr: Optional[str], output_fields: List[str],
                           metric_type: str = "IP") -> List[Any]:
    params = {"metric_type": metric_type, "params": {"ef": 80}}
    results = collection.search(
        data=[vector],
        anns_field="dense",
        param=params,
        limit=top_k,
        expr=expr or "",
        output_fields=output_fields,
    )
    return list(results[0])


# --------------------------------- Global state ----------------------------------

class MilvusContext:
    def __init__(self, host: str, port: str, collection_name: str,
                 embed_model: str, chat_model: str):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.chat_model = chat_model

        connections.connect(host=host, port=port)
        self.collection = Collection(collection_name)
        self.collection.load()
        self.text_field = pick_text_field(self.collection)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.embedder = OpenAIEmbeddings(model=embed_model, api_key=api_key)
        self.llm = ChatOpenAI(model=chat_model, api_key=api_key, temperature=0)


# ------------------------------ LangChain tool ------------------------------------

class MilvusSearchInput(BaseModel):
    query: str = Field(..., description="Refined query text to search for")
    top_k: int = Field(5, ge=1, le=1000, description="Number of results to return")
    mode: str = Field("hybrid", description="One of: semantic, keyword, hybrid")
    filter_expr: Optional[str] = Field(None, description="Milvus boolean expression for scalar filters")
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0, description="In hybrid mode, weight for keyword score")


def _milvus_search_impl(ctx: MilvusContext, args: MilvusSearchInput) -> Dict[str, Any]:
    collection = ctx.collection
    text_field = ctx.text_field

    # embed user query
    vec = ctx.embedder.embed_query(args.query)

    # candidate pool
    cand_k = max(args.top_k, 500)
    hits = milvus_semantic_search(collection, vec, cand_k, args.filter_expr, [text_field, "id"])
    cand_ids = [int(h.entity.get("id")) for h in hits]
    id_to_sem_score = {int(h.entity.get("id")): float(h.distance) for h in hits}

    # fetch text + any extra fields you commonly want
    fetched = fetch_by_ids(collection, cand_ids, fields=[text_field])

    # BM25 rerank over candidate texts
    cand_texts = [str(fetched[i][text_field]) if i in fetched else "" for i in cand_ids]
    tok_docs, idf, avgdl, k_bm25, b_bm25 = bm25_prepare(cand_texts)
    bm25_scores = bm25_score(args.query, tok_docs, idf, avgdl, k_bm25, b_bm25)

    bm25_norm = minmax_norm(bm25_scores)
    sem_scores = [id_to_sem_score.get(i, 0.0) for i in cand_ids]
    sem_norm = minmax_norm(sem_scores)

    mode = (args.mode or "hybrid").lower()
    if mode == "semantic":
        final_scores = sem_norm
        label = "semantic"
    elif mode == "keyword":
        final_scores = bm25_norm
        label = "keyword"
    else:
        w_kw = float(args.keyword_weight or 0.3)
        w_sem = 1.0 - w_kw
        final_scores = [w_sem * s + w_kw * k for s, k in zip(sem_norm, bm25_norm)]
        label = "hybrid"

    ranked = sorted(zip(cand_ids, final_scores, sem_norm, bm25_norm),
                    key=lambda x: x[1], reverse=True)[:args.top_k]

    results: List[Dict[str, Any]] = []
    for pk, s_final, s_sem, s_kw in ranked:
        row = fetched.get(pk, {"id": pk, text_field: ""})
        results.append({
            "id": pk,
            "text": row.get(text_field, ""),
            "score_final": float(s_final),
            "score_semantic": float(s_sem),
            "score_keyword": float(s_kw),
            "mode": label,
        })

    return {
        "collection": ctx.collection_name,
        "query": args.query,
        "mode": label,
        "top_k": args.top_k,
        "results": results,
    }


def make_milvus_tool(ctx: MilvusContext):
    @tool("milvus_search", args_schema=MilvusSearchInput)
    def _tool(query: str, top_k: int = 5, mode: str = "hybrid",
              filter_expr: Optional[str] = None, keyword_weight: float = 0.3) -> Dict[str, Any]:
        """Search Milvus for top_k related documents. Returns a JSON with results containing id, text, and scores.
        mode: semantic | keyword | hybrid. In hybrid, keyword_weight controls blend (0..1)."""
        args = MilvusSearchInput(query=query, top_k=top_k, mode=mode,
                                 filter_expr=filter_expr, keyword_weight=keyword_weight)
        return _milvus_search_impl(ctx, args)
    return _tool


# ------------------------------- Query refinement ----------------------------------

REFINE_PROMPT = (
    "You are a search query refiner. The user will describe their information need.\n"
    "Rewrite it into 2-3 precise search strings that target short passages. Keep them concise.\n"
    "Return them as a numbered list; no explanations."
)


def propose_queries(llm: ChatOpenAI, user_need: str) -> List[str]:
    msg = f"{REFINE_PROMPT}\n\nUser need:\n{user_need}"
    resp = llm.invoke(msg)
    text = resp.content if hasattr(resp, "content") else str(resp)
    # simple parse: lines starting with numbers
    cands: List[str] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        # remove a leading like "1) " or "1. "
        if line[0].isdigit():
            line = line.split(" ", 1)[1] if " " in line else line
            line = line.lstrip(").")
        cands.append(line.strip("- ").strip())
    # keep 3 max
    return [q for q in cands if q][:3] or [user_need]


# --------------------------------- Agent runner ------------------------------------

SYSTEM_PROMPT = (
    "You are a RAG agent with access to tools. When the user provides a refined query, "
    "you MUST call the `milvus_search` tool to retrieve documents first. Then, synthesize a concise answer "
    "grounded ONLY in the retrieved texts. Include brief bullet points and, if helpful, cite result ids."
)


def run_once(ctx: MilvusContext, agent, default_mode: str) -> bool:
    """One interactive round. Returns False to exit."""
    print("\nDescribe what you need (Enter to exit):")
    need = input("> ").strip()
    if not need:
        return False

    # Ask LLM to propose queries
    proposals = propose_queries(ctx.llm, need)
    print("\nLLM query suggestions:")
    for i, q in enumerate(proposals, 1):
        print(f"  {i}) {q}")

    choice = input("\nPick [1..{}] or edit manually: ".format(len(proposals))).strip()
    if choice.isdigit() and 1 <= int(choice) <= len(proposals):
        refined = proposals[int(choice) - 1]
    else:
        refined = choice or proposals[0]

    mode = input(f"Mode [semantic|keyword|hybrid] (default {default_mode}): ").strip().lower() or default_mode
    try:
        top_k = int(input("Top-K (default 8): ").strip() or "8")
        if top_k <= 0: top_k = 8
    except Exception:
        top_k = 8
    filt = input('Filter expr (Milvus boolean, Enter for none): ').strip() or None
    kw = 0.3
    if mode == "hybrid":
        try:
            kw_in = input("Keyword weight [0..1] (default 0.3): ").strip()
            if kw_in:
                kv = float(kw_in)
                if 0.0 <= kv <= 1.0:
                    kw = kv
        except Exception:
            pass

    user_msg = (
        f"Use milvus_search with query='{refined}', top_k={top_k}, mode='{mode}', "
        f"filter_expr={repr(filt)}, keyword_weight={kw}. Then summarize key findings."
    )

    print("\n[agent] running...")
    out = agent.invoke({"messages": [("system", SYSTEM_PROMPT), ("human", user_msg)]})

    # Agent output
    final = out["messages"][-1].content if "messages" in out else str(out)
    print("\n=== Agent Response ===\n")
    print(final)

    # Echo Top-K docs (if tool returned structured JSON in the thought)
    print("\nSearch again? [y/N]: ", end="")
    again = input().strip().lower()
    return again == "y"


def build_agent(ctx: MilvusContext, mcp_config: Optional[Dict[str, Any]]):
    tools = [make_milvus_tool(ctx)]
    if HAVE_MCP and mcp_config:
        try:
            client = MultiServerMCPClient(mcp_config)
            # Load all tools from connected servers as LangChain tools
            mcp_tools = client.get_tools()  # returns list of DynamicStructuredTool
            tools.extend(mcp_tools)
            print(f"[MCP] Loaded {len(mcp_tools)} tool(s) from MCP servers.")
        except Exception as e:
            print(f"[MCP] Skipping MCP tool loading due to error: {e}")
    else:
        if not HAVE_MCP and mcp_config:
            print("[MCP] langchain-mcp-adapters not installed; ignoring --mcp-config.")
    # Build a ReAct agent over tools
    agent = create_react_agent(ctx.llm, tools)
    return agent


def main():
    parser = argparse.ArgumentParser(description="LangGraph + MCP RAG agent for Milvus")
    parser.add_argument("--collection", required=True, help="Milvus collection name")
    parser.add_argument("--host", default=os.environ.get("MILVUS_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("MILVUS_PORT", "19530"))
    parser.add_argument("--chat-model", default="gpt-4o-mini", help="LLM chat model name")
    parser.add_argument("--embed-model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--mcp-config", default=None, help="Path to MCP servers config JSON")
    parser.add_argument("--default-mode", default="hybrid", help="Default search mode")
    args = parser.parse_args()

    # Load MCP servers config if provided
    mcp_cfg: Optional[Dict[str, Any]] = None
    if args.mcp_config:
        try:
            with open(args.mcp_config, "r", encoding="utf-8") as f:
                mcp_cfg = json.load(f)
        except Exception as e:
            print(f"[warn] Could not read MCP config: {e}. Continuing without MCP.")

    # Prepare context + agent
    ctx = MilvusContext(args.host, args.port, args.collection, args.embed_model, args.chat_model)
    agent = build_agent(ctx, mcp_cfg)

    print(f"\nConnected to Milvus '{args.collection}'  (text field: {ctx.text_field})")
    print("Human-in-the-loop RAG agent is ready.\n")
    while True:
        if not run_once(ctx, agent, args.default_mode.lower()):
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
