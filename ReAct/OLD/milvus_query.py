#!/usr/bin/env python3
"""
milvus_query_interactive.py

Interactive Milvus searcher for collections created with csv_to_milvus_4.py.

Features:
- Prompt for collection and query text
- Choose search mode: semantic, keyword, or hybrid
- Optional scalar filter expression (Milvus boolean expr)
- Configurable top_k
- Uses OpenAI embeddings for the semantic part
- Keyword / BM25-style scoring done client-side over a candidate set from Milvus

ENV:
  OPENAI_API_KEY   (required for semantic and hybrid modes)

Dependencies:
  pip install pymilvus openai pandas

Notes:
- Keyword mode does NOT rely on the collection's stored `sparse` vectors,
  so it works even if a vocab wasn't persisted at index time.
- Hybrid mode = weighted blend of semantic and keyword scores on the same candidate pool.

Tested against Milvus >= 2.3 and pymilvus >= 2.4.
"""

import os
import math
import argparse
import sys
from collections import Counter
from typing import List, Dict, Any, Tuple

import pandas as pd
from pymilvus import connections, Collection, utility
from openai import OpenAI


# ---------------- Helpers: tokenization & BM25 (client-side) ----------------
def tokenize(text: str) -> List[str]:
    return [tok for tok in ''.join(
        ch.lower() if ch.isalnum() else ' ' for ch in text or ""
    ).split() if tok]


def bm25_prepare(docs: List[str], k: float = 1.2, b: float = 0.75):
    """Precompute BM25 stats over a set of docs (candidate pool)."""
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


# ---------------- Embeddings ----------------
def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


# ---------------- Milvus search wrappers ----------------
def milvus_semantic_search(
    collection: Collection,
    vec: List[float],
    top_k: int,
    expr: str,
    output_fields: List[str],
    search_params: Dict[str, Any] = None
):
    if search_params is None:
        search_params = {"metric_type": "IP", "params": {"ef": 80}}
    results = collection.search(
        data=[vec],
        anns_field="dense",
        param=search_params,
        limit=top_k,
        expr=expr or "",
        output_fields=output_fields,
    )
    return results[0]  # single query


def fetch_texts_for_ids(
    collection: Collection,
    ids: List[int],
    text_field: str,
    output_fields: List[str]
) -> Dict[int, Dict[str, Any]]:
    """
    Fetch selected scalar fields for a set of primary key ids.
    We use a batched equality expr: id in [..].
    """
    if not ids:
        return {}
    # Milvus expr supports "in" up to a certain size; batch if needed
    out = {}
    batch = 1000
    pk_field = "id"  # csv_to_milvus_4.py uses INT64 primary key named 'id' (auto_id or not)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        expr = f"{pk_field} in {chunk}"
        rows = collection.query(expr=expr, output_fields=[text_field] + output_fields)
        for r in rows:
            out[int(r[pk_field])] = r
    return out


def pick_text_field(collection: Collection) -> str:
    """
    Heuristic: prefer 'text' if present; else first VARCHAR field other than 'dense'/'sparse'.
    """
    schema = collection.schema
    candidate = None
    for f in schema.fields:
        if f.name == "text":
            return "text"
        if getattr(f, "dtype", None).name == "VARCHAR" and f.name not in ("dense", "sparse", "id"):
            candidate = candidate or f.name
    # Fallbacks
    for f in schema.fields:
        if getattr(f, "dtype", None).name == "VARCHAR" and f.name not in ("dense", "sparse"):
            return f.name
    return "text"  # safest guess


# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Interactive Milvus query tool")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus host (default: 127.0.0.1)")
    parser.add_argument("--port", default="19530", help="Milvus port (default: 19530)")
    parser.add_argument("--openai-model", default="text-embedding-3-small",
                        help="Embedding model (must match what you used for indexing)")
    parser.add_argument("--candidate-pool", type=int, default=500,
                        help="For keyword/hybrid: how many candidates to pull from semantic search before BM25 rerank (default: 500)")
    args = parser.parse_args()

    # Connect
    connections.connect("default", host=args.host, port=args.port)

    # Choose collection
    cols = utility.list_collections() or []
    if not cols:
        print("No collections found.")
        sys.exit(1)
    print("\nAvailable collections:")
    for i, c in enumerate(cols, 1):
        print(f"  {i}) {c}")
    sel = input("\nPick a collection (number or name): ").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(cols):
        collection_name = cols[int(sel) - 1]
    else:
        collection_name = sel if sel in cols else cols[0]
        if sel not in cols:
            print(f"[info] '{sel}' not found; using '{collection_name}'")
    collection = Collection(collection_name)
    collection.load()

    # Determine likely text field
    text_field = pick_text_field(collection)
    print(f"Using text field: {text_field}")

    # Gather search inputs
    query_text = input("\nEnter your query text: ").strip()
    if not query_text:
        print("Empty query. Exiting.")
        return

    mode = input(
        "\nSearch mode? [1] semantic  [2] keyword  [3] hybrid   (default: 1): "
    ).strip() or "1"
    mode = {"1": "semantic", "2": "keyword", "3": "hybrid"}.get(mode, "semantic")

    try:
        top_k = int(input("Top-K to return (default: 10): ").strip() or "10")
        if top_k <= 0:
            top_k = 10
    except Exception:
        top_k = 10

    expr = input(
        "\nOptional Milvus scalar filter expr (e.g., year >= 2020 and category == \"news\"): "
    ).strip()

    # Which scalar fields to show?
    out_fields = input(
        "\nComma-separated extra fields to display (leave empty to auto-pick a few): "
    ).strip()
    if out_fields:
        output_fields = [f.strip() for f in out_fields.split(",") if f.strip()]
    else:
        # light defaults; you can always add more above
        output_fields = [f for f in ("title", "source", "category") if f in [fi.name for fi in collection.schema.fields]]
    # Always ensure we retrieve the primary key to rejoin
    if "id" not in output_fields:
        output_fields.append("id")

    client = None
    dense_vec = None
    if mode in ("semantic", "hybrid", "keyword"):
        # We'll still use semantic to create a candidate pool in keyword mode
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in environment.")
            sys.exit(2)
        client = OpenAI(api_key=api_key)
        dense_vec = embed_texts(client, [query_text], args.openai_model)[0]

    # ---------- Perform search ----------
    if mode == "semantic":
        hits = milvus_semantic_search(collection, dense_vec, top_k, expr, [text_field] + output_fields)
        rows = []
        for h in hits:
            rec = dict(h.entity)     # convert entity fields into a normal dict
            rec["_score_semantic"] = float(h.distance)
            rec["_rank"] = h.id
            rows.append(rec)

        df = pd.DataFrame(rows)
        # Show selected columns first
        cols_show = [c for c in ["_score_semantic", text_field] + output_fields if c in df.columns]
        df = df[cols_show]
        print("\n=== Results (semantic) ===")
        with pd.option_context("display.max_colwidth", 200):
            print(df.to_string(index=False))

    else:
        # Build a candidate pool using semantic search, then BM25-rerank
        cand_k = max(top_k, args.candidate_pool)
        hits = milvus_semantic_search(collection, dense_vec, cand_k, expr, [text_field, "id"])
        cand_ids = [int(h.entity.get("id")) for h in hits]
        id_to_sem_score = {int(h.entity.get("id")): float(h.distance) for h in hits}

        # Fetch text (+ any extra fields) for candidates
        fetched = fetch_texts_for_ids(collection, cand_ids, text_field, [f for f in output_fields if f != "id"])

        # Prepare BM25 over candidate texts
        # Keep ordering stable by cand_ids
        cand_texts = [str(fetched[i][text_field]) if i in fetched else "" for i in cand_ids]
        tok_docs, idf, avgdl, k, b = bm25_prepare(cand_texts)
        bm25_scores = bm25_score(query_text, tok_docs, idf, avgdl, k, b)

        # Normalize scores and blend if hybrid
        bm25_norm = minmax_norm(bm25_scores)
        sem_scores = [id_to_sem_score.get(i, 0.0) for i in cand_ids]
        sem_norm = minmax_norm(sem_scores)

        if mode == "keyword":
            final_scores = bm25_norm
            label = "_score_keyword"
        else:
            # hybrid weighted sum (tweak weights as needed)
            w_sem, w_kw = 0.6, 0.4
            final_scores = [w_sem * s + w_kw * k for s, k in zip(sem_norm, bm25_norm)]
            label = "_score_hybrid"

        # Rank and pick top_k
        ranked = sorted(
            zip(cand_ids, final_scores, sem_norm, bm25_norm),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        rows = []
        for pk, score, s_sem, s_kw in ranked:
            rec = fetched.get(pk, {"id": pk, text_field: ""})
            rec = dict(rec)  # copy
            rec[label] = float(score)
            rec["_score_semantic_norm"] = float(s_sem)
            rec["_score_keyword_norm"] = float(s_kw)
            rows.append(rec)

        df = pd.DataFrame(rows)
        cols_show = [c for c in [label, "_score_semantic_norm", "_score_keyword_norm", text_field] + output_fields if c in df.columns]
        df = df[cols_show]
        print("\n=== Results (semantic) ===")
        with pd.option_context("display.max_colwidth", 200):
            print(df.to_string(index=False))

        print("\n=== Top-K Related Documents ===")
        for i, row in df.iterrows():
            doc_text = str(row.get(text_field, ""))[:500]  # trim for readability
            print(f"\n[{i+1}] {doc_text}")



if __name__ == "__main__":
    main()
