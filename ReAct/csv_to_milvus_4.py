# csv_to_milvus_4.py
import os
import math
import argparse
from collections import Counter
from pathlib import Path
import glob

import pandas as pd
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from openai import OpenAI


# ---------------- Helpers ----------------
def pick_text_column(frame: pd.DataFrame) -> str:
    if "text" in frame.columns:
        return "text"
    for col in frame.columns:
        if frame[col].dtype == "object":
            sample = frame[col].dropna().astype(str).str.strip()
            if (sample != "").any():
                return col
    return frame.columns[0]


def embed_texts(client, texts, model):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def tokenize(text):
    return [tok for tok in ''.join(
        ch.lower() if ch.isalnum() else ' ' for ch in text
    ).split() if tok]


def build_bm25_sparse(texts, k=1.2, b=0.75):
    tokenized = [tokenize(t) for t in texts]
    N = len(tokenized) or 1
    avgdl = (sum(len(toks) for toks in tokenized) / N) if N else 0.0

    dfreq = Counter()
    for toks in tokenized:
        dfreq.update(set(toks))

    vocab = {term: i for i, term in enumerate(sorted(dfreq.keys()))}
    idf = {t: math.log((N - dfreq[t] + 0.5) / (dfreq[t] + 0.5) + 1) for t in dfreq}

    sparse_vecs = []
    for toks in tokenized:
        tf = Counter(toks)
        dl = len(toks) or 1
        vec = {}
        for term, f in tf.items():
            tid = vocab[term]
            w = idf[term] * ((f * (k + 1)) / (f + k * (1 - b + b * dl / avgdl)))
            if w > 0:
                vec[tid] = float(w)
        sparse_vecs.append(vec)

    return sparse_vecs


def is_unique_int_series(s: pd.Series) -> bool:
    try:
        si = pd.to_numeric(s, errors="coerce").dropna().astype("int64")
        return (len(si) == len(s)) and si.is_unique
    except Exception:
        return False


def compute_varchar_len(s: pd.Series, cap: int = 4096, default: int = 256) -> int:
    try:
        lengths = s.fillna("").astype(str).map(len)
        m = int(lengths.max()) if len(lengths) else default
        if m <= 128: return 128
        if m <= 256: return 256
        if m <= 512: return 512
        if m <= 1024: return 1024
        if m <= 2048: return 2048
        return min(cap, 4096)
    except Exception:
        return default


# ---------------- Main -------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load a CSV into Milvus, automatically inferring schema from the header."
    )
    # make csv/collection optional so we can prompt the user
    parser.add_argument("--csv", help="Path to input CSV file (will prompt if omitted)")
    parser.add_argument("--collection", help="Milvus collection name (will prompt if omitted)")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus server host (default: 127.0.0.1)")
    parser.add_argument("--port", default="19530", help="Milvus server port (default: 19530)")
    parser.add_argument("--openai-model", default="text-embedding-3-small", help="OpenAI embedding model to use")
    args = parser.parse_args()

    EMBED_DIM = 1536  # fixed for text-embedding-3-small
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Connect
    connections.connect("default", host=args.host, port=args.port)

    # 1) List existing collections
    try:
        existing = utility.list_collections() or []
        if existing:
            print("Existing Milvus collections:")
            for i, c in enumerate(existing, 1):
                print(f"  {i}) {c}")
            # Offer interactive deletion after listing
            sel = input("\nDelete any collections? Enter numbers (e.g. 1,3), names (comma-separated), 'all' to drop all, or press Enter to keep: ").strip()
            if sel:
                to_delete = []
                if sel.lower() == "all":
                    confirm_all = input("Confirm drop ALL collections listed above? This is irreversible. [y/N]: ").strip().lower()
                    if confirm_all == "y":
                        to_delete = list(existing)
                else:
                    parts = [s.strip() for s in sel.split(",") if s.strip()]
                    for p in parts:
                        if p.isdigit():
                            idx = int(p) - 1
                            if 0 <= idx < len(existing):
                                to_delete.append(existing[idx])
                        else:
                            # treat as name if it matches
                            if p in existing:
                                to_delete.append(p)
                # perform deletions with per-collection confirmation
                for name in to_delete:
                    ok = input(f"Confirm drop collection '{name}'? [y/N]: ").strip().lower()
                    if ok == "y":
                        try:
                            utility.drop_collection(name)
                            print(f"Dropped collection '{name}'.")
                        except Exception as e:
                            print(f"[warn] failed to drop '{name}': {e}")
                    else:
                        print(f"Skipped dropping '{name}'.")
        else:
            print("No existing Milvus collections found.")
    except Exception as e:
        print("[warn] could not list collections:", e)

    # 2) Ask for CSV file path (prompt if not provided). Validate file exists.
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else None
    while csv_path is None or not csv_path.exists() or not csv_path.is_file():
        candidate = input("\nEnter path to CSV file: ").strip()
        if not candidate:
            continue
        p = Path(candidate).expanduser().resolve()
        if not p.exists():
            print("File not found:", p)
            continue
        if not p.is_file():
            print("Not a file:", p)
            continue
        csv_path = p
    print(f"\nUsing CSV: {csv_path}")

    # 3) Ask for collection name (prompt if not provided). Offer to drop/recreate if exists.
    collection_name = args.collection or ""
    while not collection_name:
        collection_name = input("\nEnter name for the new Milvus collection: ").strip()
    if utility.has_collection(collection_name):
        resp = input(f"Collection '{collection_name}' already exists. Drop and recreate? [y/N]: ").strip().lower()
        if resp == "y":
            try:
                utility.drop_collection(collection_name)
                print(f"Dropped collection '{collection_name}'.")
            except Exception as e:
                print(f"[warn] failed to drop collection '{collection_name}': {e}")
        else:
            print(f"Reusing existing collection '{collection_name}'.")

    # Load CSV
    df = pd.read_csv(csv_path)
    print("CSV preview:")
    print(df.head())

    # Pick text col
    TEXT_COL = pick_text_column(df)
    print(f"Embedding text column: {TEXT_COL!r}")

    # Generate embeddings
    dense_vecs = embed_texts(client, df[TEXT_COL].astype(str).fillna("").tolist(), args.openai_model)

    # Sparse vectors
    sparse_vecs = build_bm25_sparse(df[TEXT_COL].astype(str).fillna("").tolist())

    # Schema inference
    auto_id = True
    from pymilvus import FieldSchema, DataType

    if "id" in df.columns and is_unique_int_series(df["id"]):
        auto_id = False
        primary_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    else:
        primary_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)

    field_schemas = [primary_field]
    for col in df.columns:
        if col == "id" and auto_id:
            print("Auto ID enabled; ignoring user 'id' column.")
            continue
        if col == "id" and not auto_id:
            field_schemas.append(FieldSchema(name=col, dtype=DataType.INT64))
            continue

        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            fs = FieldSchema(name=col, dtype=DataType.BOOL)
        elif pd.api.types.is_integer_dtype(s):
            fs = FieldSchema(name=col, dtype=DataType.INT64)
        elif pd.api.types.is_float_dtype(s):
            fs = FieldSchema(name=col, dtype=DataType.DOUBLE)
        else:
            max_len = compute_varchar_len(s)
            fs = FieldSchema(name=col, dtype=DataType.VARCHAR, max_length=max_len)
        field_schemas.append(fs)

    field_schemas.append(FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM))
    field_schemas.append(FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR))

    # Create collection
    # Use the top-level pymilvus imports (imported at module top). Do not re-import inside the function.
    if utility.has_collection(collection_name):
        # if user opted to reuse existing collection we skip create; otherwise it's dropped above
        print(f"Using existing collection: {collection_name}")
        collection = Collection(collection_name)
    else:
        schema = CollectionSchema(field_schemas, description="Auto-schema from CSV header")
        collection = Collection(collection_name, schema)

    # Indexes
    collection.create_index("dense", {
        "index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}
    })
    collection.create_index("sparse", {
        "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "params": {}
    })

    # Insert
    insert_payload = []
    for fs in field_schemas:
        if fs.name == "id" and fs.is_primary and fs.auto_id:
            continue
        elif fs.name == "dense":
            insert_payload.append(dense_vecs)
        elif fs.name == "sparse":
            insert_payload.append(sparse_vecs)
        else:
            col_series = df[fs.name]
            if fs.dtype == DataType.VARCHAR:
                insert_payload.append(col_series.fillna("").astype(str).tolist())
            elif fs.dtype == DataType.INT64:
                insert_payload.append(pd.to_numeric(col_series, errors="coerce").fillna(0).astype("int64").tolist())
            elif fs.dtype == DataType.DOUBLE:
                insert_payload.append(pd.to_numeric(col_series, errors="coerce").fillna(0.0).astype("float64").tolist())
            elif fs.dtype == DataType.BOOL:
                vals = col_series.fillna(False)
                if vals.dtype != bool:
                    vals = vals.astype(str).str.lower().isin(["1", "true", "yes", "y", "t"])
                insert_payload.append(vals.astype(bool).tolist())
            else:
                insert_payload.append(col_series.fillna("").astype(str).tolist())

    mr = collection.insert(insert_payload)
    collection.flush()
    collection.load()

    print(f"Inserted {len(df)} rows into '{collection_name}'. Auto ID = {auto_id}")


if __name__ == "__main__":
    main()
