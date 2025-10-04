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
    # make collection/folder optional so we can prompt interactively
    p.add_argument("--collection", help="Milvus collection name (will prompt if omitted)")
    p.add_argument("--folder", help="Folder containing CSV files (will prompt if omitted)")
    p.add_argument("--milvus-uri", help="Milvus server URI, e.g. http://localhost:19530")
    p.add_argument("--milvus-token", help="Milvus/Zilliz Cloud token, often 'username:password'", default=None)
    p.add_argument("--openai-model", help="OpenAI embedding model (default: text-embedding-3-large)")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for embeddings")
    p.add_argument("--id-column", default="id", help="Name of the ID column in CSV (default: id)")
    p.add_argument("--text-column", default="text", help="Name of the text column in CSV (default: text)")

    # Sparse/hybrid configuration: enabled by default, use --no-sparse to disable
    p.add_argument("--enable-sparse", action="store_true", default=True,
                   help="Add nullable SPARSE_FLOAT_VECTOR field 'sparse' (enabled by default)")
    p.add_argument("--no-sparse", action="store_false", dest="enable_sparse",
                   help="Disable creation of the 'sparse' field")
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


def ensure_collection(
    name: str,
    dim: int,
    enable_sparse: bool,
    sparse_index_type: Optional[str],
) -> Collection:
    """
    Ensure collection exists. If enable_sparse is True, ensures a nullable 'sparse' field is present.
    This function will not drop/recreate; that logic is handled by the caller in main().
    """
    if utility.has_collection(name):
        # Collection exists, just return it. Schema validation is done by the caller.
        col = Collection(name)
        # When reusing a collection, we must still load it.
        # We need to check if the sparse index exists to avoid load errors.
        has_sparse_index = False
        if _has_field(col, "sparse"):
            try:
                if any(idx.field_name == "sparse" for idx in col.indexes):
                    has_sparse_index = True
            except Exception:
                pass  # best-effort check
        
        if _has_field(col, "sparse") and not has_sparse_index:
            print("[warn] Reusing collection with un-indexed 'sparse' field. Loading dense 'embedding' field only.")
            col.load(field_names=["embedding"])
        else:
            col.load()
        return col

    # --- Create collection if it doesn't exist ---
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="meta", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    if enable_sparse:
        # Add a nullable sparse vector field. is_nullable=True is critical.
        fields.append(
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR, is_nullable=True)
        )

    schema = CollectionSchema(fields=fields, description="CSV -> Milvus with dense/sparse embeddings")
    col = Collection(name=name, schema=schema)
    print(f"Created new collection '{name}' with schema: {[f.name for f in fields]}")

    # Create index for the dense vector field
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64},
    }
    col.create_index(field_name="embedding", index_params=index_params)

    # If sparse field was added, attempt to create a sparse index
    sparse_index_created = False
    if enable_sparse:
        # Try a list of common sparse index types.
        # If a specific type is provided via CLI, try that one first.
        types_to_try = []
        if sparse_index_type:
            types_to_try.append(sparse_index_type)
        types_to_try.extend(["SPARSE_INVERTED_INDEX", "BM25"]) # Add other common types if needed

        for index_type in set(types_to_try): # Use set to avoid duplicates
            try:
                print(f"[info] Attempting to create sparse index with type: {index_type}...")
                col.create_index(field_name="sparse", index_params={"index_type": index_type})
                print(f"[info] Successfully created sparse index on 'sparse' field with type: {index_type}")
                sparse_index_created = True
                break  # Stop after the first success
            except Exception as e:
                print(f"[warn] Could not create sparse index with type '{index_type}': {e}")
        
        if not sparse_index_created:
            print("[warn] Failed to create a sparse index with any of the attempted types.")
            print("[warn] You can specify a different supported type with --sparse-index-type.")

    # Load the collection, being careful about un-indexed sparse fields.
    if enable_sparse and not sparse_index_created:
        print("[warn] Sparse field exists but is not indexed. Loading dense 'embedding' field only.")
        col.load(field_names=["embedding"])
    else:
        col.load()
        
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
    milvus_uri = args.milvus_uri or input(f"Milvus server URI [{default_uri}]: ").strip() or default_uri
    milvus_token = args.milvus_token or os.getenv("MILVUS_TOKEN")
    print(f"Connecting to Milvus at {milvus_uri} ...")
    connections.connect(alias="default", uri=milvus_uri, token=milvus_token)

    try:
        cols = utility.list_collections()
        if cols:
            print("Existing collections:", ", ".join(cols))
    except Exception:
        pass

    # Prompt for collection name if not provided
    collection = args.collection or ""
    while not collection:
        collection = input("\nEnter Milvus collection name to create/use: ").strip()

    # Check existing collection and decide if it needs to be dropped
    try:
        if utility.has_collection(collection):
            col = Collection(collection)
            sparse_field = next((f for f in col.schema.fields if f.name == "sparse"), None)
            
            # Check for the problematic schema: sparse field exists but is not nullable
            if args.enable_sparse and (sparse_field is None or not getattr(sparse_field, 'is_nullable', False)):
                print(f"\n[!] Collection '{collection}' exists but is missing a nullable 'sparse' field.")
                resp = input("Drop and recreate the collection to fix the schema? This will delete all data. [y/N]: ").strip().lower()
                if resp == 'y':
                    utility.drop_collection(collection)
                    print(f"Dropped collection '{collection}'. It will be recreated.")
                else:
                    print("[ERROR] Aborting. The script cannot proceed without a nullable 'sparse' field.", file=sys.stderr)
                    sys.exit(1)
            else:
                 print(f"Using existing collection '{collection}'.")

    except Exception as e:
        print(f"[warn] Could not inspect existing collection, proceeding with creation logic: {e}")
        pass

    # Prompt for folder if not provided
    folder_arg = args.folder or ""
    while not folder_arg:
        folder_arg = input("Enter folder path with CSV files: ").strip()
    folder = Path(folder_arg).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found or not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    openai_model = args.openai_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    embedder = Embedder(openai_model)
    print(f"Probing embedding dimension for model '{openai_model}' ...")
    dim = embedder.infer_dim()
    print(f"Embedding dimension = {dim}")

    # Ensure collection (if we decided to recreate above, enable_sparse will be honoured here)
    col = ensure_collection(
        name=collection,
        dim=dim,
        enable_sparse=args.enable_sparse,
        sparse_index_type=args.sparse_index_type,
    )
    print(f"Using collection: {col.name}")

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

    # Flush / persist
    col.flush()

    print("\nSummary")
    print("-------")
    print(f"Collection: {collection}")
    print(f"CSV files:  {len(csvs)}")
    print(f"Inserted:   {total_inserted} rows")
    print("Done.")


if __name__ == "__main__":
    main()
