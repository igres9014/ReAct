#!/usr/bin/env python3
"""
CSV ➜ Milvus loader with OpenAI embeddings (dense-only by default)

Key changes vs. your current script:
- **Dense-only by default**: drops any "sparse" field and **does not** create a sparse index unless explicitly enabled.
- **Opt-in sparse**: use `--enable-sparse` (nullable) and optionally `--sparse-index-type` (if your Milvus build supports it).
- **No 'sparse' required on insert**: avoids "Insert missed field `sparse`" errors.
- **Safe load**: loads only the 'embedding' field unless sparse is enabled and indexed.

Usage examples
--------------
# Dense-only (recommended)
python csv_to_milvus_fixed.py --collection PROVA_X7 --folder ./my_csvs --milvus-uri http://localhost:19530

# If you know your Milvus supports sparse index types (check docs/build) and you want hybrid:
python csv_to_milvus_fixed.py --collection PROVA_X7 --folder ./my_csvs --enable-sparse --sparse-index-type BM25
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Iterable, Optional

import pandas as pd
from tqdm import tqdm

# OpenAI SDK v1 style
try:
    from openai import OpenAI
except ImportError as e:
    print("The 'openai' package is required. Install with: pip install openai", file=sys.stderr)
    raise

# Milvus / PyMilvus
try:
    from pymilvus import (
        connections,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        utility,
    )
except ImportError:
    print("The 'pymilvus' package is required. Install with: pip install pymilvus", file=sys.stderr)
    raise


@dataclass
class Config:
    collection: str
    folder: Path
    milvus_uri: str
    milvus_token: str | None
    openai_model: str
    batch_size: int
    id_column: str
    text_column: str
    enable_sparse: bool
    sparse_index_type: Optional[str]


# ----------------------------- CLI ----------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load CSV rows into Milvus with OpenAI embeddings (dense-first)")
    p.add_argument("--collection", required=True, help="Milvus collection name")
    p.add_argument("--folder", required=True, help="Folder containing CSV files")
    p.add_argument("--milvus-uri", help="Milvus server URI, e.g. http://localhost:19530")
    p.add_argument("--milvus-token", help="Milvus/Zilliz Cloud token, often 'username:password'", default=None)
    p.add_argument("--openai-model", help="OpenAI embedding model (default: text-embedding-3-large)")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings")
    p.add_argument("--id-column", default="id", help="Name of the ID column in CSV (default: id)")
    p.add_argument("--text-column", default="text", help="Name of the text column in CSV (default: text)")

    # Sparse/hybrid configuration (opt-in)
    p.add_argument("--enable-sparse", action="store_true", help="Add nullable SPARSE_FLOAT_VECTOR field 'sparse' (opt-in)")
    p.add_argument("--sparse-index-type", default=None, help="Index type for sparse field if enabled (e.g., BM25)")

    return p.parse_args()


# ----------------------------- OpenAI helpers ---------------------------- #

class Embedder:
    def __init__(self, model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter OPENAI_API_KEY: ").strip()
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dim_cache: int | None = None

    def infer_dim(self) -> int:
        if self._dim_cache:
            return self._dim_cache
        resp = self.client.embeddings.create(model=self.model, input=["test"])
        dim = len(resp.data[0].embedding)
        self._dim_cache = dim
        return dim

    def embed_batch(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        for attempt in range(max_retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as e:
                wait = 2 ** attempt
                print(f"Embedding batch failed (attempt {attempt+1}): {e}. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError("Failed to embed batch after retries")


# ----------------------------- Milvus helpers ---------------------------- #

def _has_field(col: Collection, field_name: str) -> bool:
    try:
        return any(f.name == field_name for f in col.schema.fields)
    except Exception:
        return False


def ensure_collection(name: str, dim: int, enable_sparse: bool, sparse_index_type: Optional[str]) -> Collection:
    if utility.has_collection(name):
        col = Collection(name)
        if _has_field(col, "sparse"):
            print("[warn] Existing collection has a 'sparse' field.")
        return col

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="meta", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    if enable_sparse:
        fields.append(FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR, is_nullable=True))

    schema = CollectionSchema(fields=fields, description="CSV -> Milvus (dense-first)")
    col = Collection(name=name, schema=schema)

    col.create_index(
        field_name="embedding",
        index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}},
    )

    if enable_sparse and sparse_index_type:
        try:
            col.create_index(field_name="sparse", index_params={"index_type": sparse_index_type, "params": {}})
            print(f"[info] created sparse index on 'sparse' with index_type={sparse_index_type}")
        except Exception as e:
            print(f"[warn] could not create sparse index '{sparse_index_type}' on 'sparse': {e}")

    try:
        if enable_sparse and sparse_index_type:
            col.load()
        else:
            try:
                col.load(field_names=["embedding"])
            except TypeError:
                col.load()
    except Exception as e:
        print("[warn] load warning:", e)

    return col


def insert_rows(col: Collection, rows: Iterable[Dict[str, Any]], batch_size: int = 64):
    batch: List[Dict[str, Any]] = []
    count = 0
    for r in rows:
        if "sparse" in r and r["sparse"] is None:
            r.pop("sparse")
        batch.append(r)
        if len(batch) >= batch_size:
            col.insert(batch)
            count += len(batch)
            batch.clear()
    if batch:
        col.insert(batch)
        count += len(batch)
    return count


# ----------------------------- CSV ingestion ----------------------------- #

def discover_csvs(folder: Path) -> List[Path]:
    files = sorted([p for p in folder.glob("**/*.csv") if p.is_file()])
    if not files:
        print(f"No CSV files found under {folder}")
    return files


def iter_rows_from_df(df: pd.DataFrame, id_col: str, text_col: str) -> Iterable[Dict[str, Any]]:
    if text_col not in df.columns:
        raise ValueError(f"CSV is missing required text column '{text_col}'. Columns: {list(df.columns)}")

    df[text_col] = df[text_col].astype(str).fillna("")

    has_id = id_col in df.columns
    meta_cols = [c for c in df.columns if c not in {id_col, text_col}]

    for idx, row in df.iterrows():
        rid = str(row[id_col]) if has_id else str(uuid.uuid5(uuid.NAMESPACE_URL, f"row/{idx}"))
        text = str(row[text_col])
        meta = {c: (None if pd.isna(row[c]) else row[c]) for c in meta_cols}
        yield {"id": rid, "text": text, "meta": meta}


def embed_and_insert(embedder: Embedder, col: Collection, df: pd.DataFrame, id_col: str, text_col: str, batch_size: int):
    rows_iter = list(iter_rows_from_df(df, id_col, text_col))
    texts = [r["text"] for r in rows_iter]

    total = len(rows_iter)
    if total == 0:
        return 0

    inserted = 0
    for start in tqdm(range(0, total, batch_size), desc="Embedding + inserting", unit="rows"):
        end = min(start + batch_size, total)
        batch_rows = rows_iter[start:end]
        batch_texts = texts[start:end]
        vecs = embedder.embed_batch(batch_texts)
        for r, v in zip(batch_rows, vecs):
            r["embedding"] = v
        inserted += insert_rows(col, batch_rows, batch_size=batch_size)
    return inserted


# ----------------------------- Main flow -------------------------------- #

def main():
    args = parse_args()

    default_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_uri = args.milvus_uri or default_uri
    milvus_token = args.milvus_token or os.getenv("MILVUS_TOKEN")
    print(f"Connecting to Milvus at {milvus_uri} ...")
    connections.connect(alias="default", uri=milvus_uri, token=milvus_token)

    try:
        cols = utility.list_collections()
        if cols:
            print("Existing collections:", ", ".join(cols))
    except Exception:
        pass

    openai_model = args.openai_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    embedder = Embedder(openai_model)
    print(f"Probing embedding dimension for model '{openai_model}' ...")
    dim = embedder.infer_dim()
    print(f"Embedding dimension = {dim}")

    col = ensure_collection(
        name=args.collection,
        dim=dim,
        enable_sparse=args.enable_sparse,
        sparse_index_type=args.sparse_index_type,
    )
    print(f"Using collection: {col.name}")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found or not a directory: {folder}", file=sys.stderr)
        sys.exit(1)
    csvs = discover_csvs(folder)
    if not csvs:
        sys.exit(1)

    total_inserted = 0
    for csv_path in csvs:
        print(f"\nReading {csv_path} ...")
        df = pd.read_csv(csv_path)
        inserted = embed_and_insert(embedder, col, df, args.id_column, args.text_column, args.batch_size)
        print(f"Inserted {inserted} rows from {csv_path.name}")
        total_inserted += inserted

    try:
        col.flush()
    except Exception:
        pass

    print("\nSummary")
    print("-------")
    print(f"Collection: {args.collection}")
    print(f"CSV files:  {len(csvs)}")
    print(f"Inserted:   {total_inserted} rows")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
