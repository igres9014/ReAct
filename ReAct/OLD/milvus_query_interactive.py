#!/usr/bin/env python3
"""
milvus_query_interactive.py  (v3)

Interactive Milvus search tool for collections created from CSVs (e.g., via csv_to_milvus_4.py).

Features:
- Prompt for collection and query text
- Choose search mode: semantic, keyword (BM25-like), or hybrid
- Optional scalar filter (Milvus expr)
- Configurable top_k
- Uses OpenAI embeddings for semantic search
- Client-side BM25 rerank for keyword/hybrid
- Prints a Top-K Related Documents section (always)
- **NEW**: User-adjustable keyword score weight for HYBRID mode (0..1, default 0.3)

ENV:
  OPENAI_API_KEY   (required for semantic/keyword/hybrid since we use embeddings to create a candidate pool)

Dependencies:
  pip install pymilvus openai pandas

Tested with:
  Milvus >= 2.3
  pymilvus >= 2.4
"""

import os
import math
import argparse
import sys
from collections import Counter
from typing import List, Dict, Any

import pandas as pd
from pymilvus import connections, Collection, utility
from openai import OpenAI


# ------------------------------- Tokenization & BM25 -------------------------------

def tokenize(text: str) -> List[str]:
    return [tok for tok in ''.join(
        ch.lower() if ch.isalnum() else ' ' for ch in (text or "")
    ).split() if tok]


def bm25_prepare(docs: List[str], k: float = 1.2, b: float = 0.75):
    """Precompute BM25 stats over a set of docs (candidate pool)."""
    tokenized = [tokenize(t) for t in docs]
    N = len(tokenized) or 1
    avgdl = (sum(len(toks) for toks in tokenized) / N) if N else 0.0

    dfreq = Counter()
    for toks in tokenized:
        dfreq.update(set(toks))

    # standard BM25 idf
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


# ------------------------------------ Embeddings -----------------------------------

def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


# -------------------------------- Milvus helpers -----------------------------------

def milvus_semantic_search(
    collection: Collection,
    vec: List[float],
    top_k: int,
    expr: str,
    output_fields: List[str],
    search_params: Dict[str, Any] = None
):
    """Wrapper around collection.search for a single query vector."""
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
    return results[0]  # we passed a single query vector


def fetch_texts_for_ids(
    collection: Collection,
    ids: List[int],
    text_field: str,
    extra_fields: List[str],
    id_col: str = "id",
) -> Dict[int, Dict[str, Any]]:
    """
    Fetch selected scalar fields for a set of primary key ids: id + text_field + extra_fields.
    """
    if not ids:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    batch = 1000
    fields_to_get = [id_col, text_field] + [f for f in extra_fields if f not in (id_col, text_field)]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        expr = f"{id_col} in {chunk}"
        rows = collection.query(expr=expr, output_fields=fields_to_get)
        for r in rows:
            out[int(r[id_col])] = r
    return out


def pick_text_field(collection: Collection) -> str:
    """
    Heuristic: prefer 'text' if present; else first VARCHAR-like field other than dense/sparse/id.
    """
    schema = collection.schema
    # First pass: explicit 'text'
    for f in schema.fields:
        if getattr(f, "name", None) == "text":
            return "text"

    # Otherwise find a string field
    def is_varchar(field) -> bool:
        dtype_name = getattr(field, "dtype", None)
        dtype_name = getattr(dtype_name, "name", str(dtype_name))
        return str(dtype_name).upper() in ("VARCHAR", "STRING")

    for f in schema.fields:
        if is_varchar(f) and f.name not in ("dense", "sparse", "id"):
            return f.name

    # Last resort
    return "text"


# ------------------------------- Output helpers ------------------------------------

def ensure_text_field_in_df(df, collection, text_field, id_col="id"):
    """
    Ensure df includes the text_field. If missing, fetch it by ids from Milvus and merge.
    """
    if text_field in df.columns:
        return df

    if id_col not in df.columns or df.empty:
        return df  # nothing we can do

    ids = [int(i) for i in df[id_col].tolist() if pd.notna(i)]
    if not ids:
        return df

    # fetch text in batches
    fetched_map = {}
    batch = 1000
    fields_to_get = [id_col, text_field]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        expr = f"{id_col} in {chunk}"
        rows = collection.query(expr=expr, output_fields=fields_to_get)
        for r in rows:
            fetched_map[int(r[id_col])] = r.get(text_field, "")

    if not fetched_map:
        return df

    text_series = df[id_col].map(lambda x: fetched_map.get(int(x), "") if pd.notna(x) else "")
    df = df.copy()
    df[text_field] = text_series
    return df


def print_top_k_documents(df, text_field, limit_chars=500, header="=== Top-K Related Documents ==="):
    """
    Pretty-print the top-K docs from the current df (in displayed order).
    """
    print(f"\n{header}")
    if df.empty:
        print("\n[no results]")
        return
    for idx, row in df.reset_index(drop=True).iterrows():
        doc_text = str(row.get(text_field, ""))
        if limit_chars and limit_chars > 0:
            doc_text = doc_text[:limit_chars]
        print(f"\n[{idx+1}] {doc_text}")


# -------------------------------------- CLI ----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive Milvus query tool")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus host (default: 127.0.0.1)")
    parser.add_argument("--port", default="19530", help="Milvus port (default: 19530)")
    parser.add_argument("--openai-model", default="text-embedding-3-small",
                        help="Embedding model (should match what you used for indexing)")
    parser.add_argument("--candidate-pool", type=int, default=500,
                        help="For keyword/hybrid: number of semantic candidates to rerank (default: 500)")
    parser.add_argument("--show-doc-chars", type=int, default=500,
                        help="Max characters to display per document (default: 500)")
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

    # Gather inputs
    query_text = input("\nEnter your query text: ").strip()
    if not query_text:
        print("Empty query. Exiting.")
        return

    mode = input("\nSearch mode? [1] semantic  [2] keyword  [3] hybrid   (default: 1): ").strip() or "1"
    mode = {"1": "semantic", "2": "keyword", "3": "hybrid"}.get(mode, "semantic")

    # NEW: keyword score weight for hybrid mode
    keyword_weight = 0.3
    if mode == "hybrid":
        try:
            kw_in = input("Keyword score weight [0-1] (default: 0.3): ").strip()
            if kw_in:
                kw_val = float(kw_in)
                if 0.0 <= kw_val <= 1.0:
                    keyword_weight = kw_val
                else:
                    print("[info] Invalid weight, using default 0.3")
        except Exception:
            print("[info] Could not parse weight, using default 0.3")

    try:
        top_k = int(input("Top-K to return (default: 10): ").strip() or "10")
        if top_k <= 0:
            top_k = 10
    except Exception:
        top_k = 10

    expr = input('\nOptional Milvus scalar filter expr (e.g., year >= 2020 and category == "news"): ').strip()

    # Which scalar fields to show?
    out_fields = input("\nComma-separated extra fields to display (leave empty to auto-pick a few): ").strip()
    if out_fields:
        output_fields = [f.strip() for f in out_fields.split(",") if f.strip()]
    else:
        # auto-pick a few popular names if they exist
        existing = [fi.name for fi in collection.schema.fields]
        output_fields = [f for f in ("title", "source", "category") if f in existing]
    # Always ensure we retrieve the primary key to rejoin
    if "id" not in output_fields:
        output_fields.append("id")

    # Embedding client (we also use semantic search to form a candidate pool in keyword mode)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment.")
        sys.exit(2)
    client = OpenAI(api_key=api_key)
    dense_vec = embed_texts(client, [query_text], args.openai_model)[0]

    # --------------------------------- SEMANTIC ---------------------------------
    if mode == "semantic":
        hits = milvus_semantic_search(
            collection,
            dense_vec,
            top_k,
            expr,
            output_fields=[text_field] + output_fields
        )

        rows = []
        for h in hits:
            # h.entity is dict-like; convert to plain dict
            rec = dict(h.entity)
            rec["_score_semantic"] = float(h.distance)
            rec["_rank"] = h.id
            rows.append(rec)

        df = pd.DataFrame(rows)

        # Guarantee the text field exists (safety net)
        df = ensure_text_field_in_df(df, collection, text_field, id_col="id")

        # Show table
        cols_show = [c for c in ["_score_semantic", text_field] + output_fields if c in df.columns]
        if cols_show:
            df = df[cols_show]
        print("\n=== Results (semantic) ===")
        with pd.option_context("display.max_colwidth", 200):
            print(df.to_string(index=False) if not df.empty else "[no results]")

        # Print Top-K docs
        print_top_k_documents(df, text_field, limit_chars=args.show_doc_chars)

    # ------------------------------ KEYWORD / HYBRID ------------------------------
    else:
        cand_k = max(top_k, args.candidate_pool)
        # Pull a candidate pool via semantic search (ids + text)
        hits = milvus_semantic_search(
            collection,
            dense_vec,
            cand_k,
            expr,
            output_fields=[text_field, "id"]
        )
        cand_ids = [int(h.entity.get("id")) for h in hits]
        id_to_sem_score = {int(h.entity.get("id")): float(h.distance) for h in hits}

        # Fetch text (+ any extra fields) for candidates
        fetched = fetch_texts_for_ids(
            collection,
            cand_ids,
            text_field,
            extra_fields=[f for f in output_fields if f != "id"],
            id_col="id",
        )

        # Prepare BM25 over candidate texts preserving cand_ids order
        cand_texts = [str(fetched[i][text_field]) if i in fetched else "" for i in cand_ids]
        tok_docs, idf, avgdl, k_bm25, b_bm25 = bm25_prepare(cand_texts)
        bm25_scores = bm25_score(query_text, tok_docs, idf, avgdl, k_bm25, b_bm25)

        # Normalize scores and blend (hybrid) or use keyword only
        bm25_norm = minmax_norm(bm25_scores)
        sem_scores = [id_to_sem_score.get(i, 0.0) for i in cand_ids]
        sem_norm = minmax_norm(sem_scores)

        if mode == "keyword":
            final_scores = bm25_norm
            label = "_score_keyword"
        else:
            # hybrid weighted sum using user-selected keyword weight
            w_kw = float(keyword_weight)
            w_sem = 1.0 - w_kw
            final_scores = [w_sem * s + w_kw * k for s, k in zip(sem_norm, bm25_norm)]
            label = "_score_hybrid"

        ranked = sorted(
            zip(cand_ids, final_scores, sem_norm, bm25_norm),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        rows = []
        for pk, score, s_sem, s_kw in ranked:
            rec = fetched.get(pk, {"id": pk, text_field: ""})
            rec = dict(rec)
            rec[label] = float(score)
            rec["_score_semantic_norm"] = float(s_sem)
            rec["_score_keyword_norm"] = float(s_kw)
            rows.append(rec)

        df = pd.DataFrame(rows)

        # Guarantee text field is present
        df = ensure_text_field_in_df(df, collection, text_field, id_col="id")

        # Show table
        cols_show = [c for c in [label, "_score_semantic_norm", "_score_keyword_norm", text_field] + output_fields if c in df.columns]
        if cols_show:
            df = df[cols_show]
        print(f"\n=== Results ({mode}) ===")
        with pd.option_context("display.max_colwidth", 200):
            print(df.to_string(index=False) if not df.empty else "[no results]")

        # Print Top-K docs
        print_top_k_documents(df, text_field, limit_chars=args.show_doc_chars)


if __name__ == "__main__":
    main()
