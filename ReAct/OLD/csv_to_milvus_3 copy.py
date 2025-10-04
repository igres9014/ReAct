# csv_to_milvus_3.py
import os
import math
import argparse
from collections import Counter

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
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--collection", default="csv_docs_with_metadata", help="Milvus collection name")
    parser.add_argument("--host", default="127.0.0.1", help="Milvus server host (default: 127.0.0.1)")
    parser.add_argument("--port", default="19530", help="Milvus server port (default: 19530)")
    parser.add_argument("--openai-model", default="text-embedding-3-small", help="OpenAI embedding model to use")
    args = parser.parse_args()

    EMBED_DIM = 1536  # fixed for text-embedding-3-small
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Connect
    connections.connect("default", host=args.host, port=args.port)

    # Load CSV
    df = pd.read_csv(args.csv)
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
    from pymilvus import CollectionSchema, Collection, utility
    if utility.has_collection(args.collection):
        utility.drop_collection(args.collection)

    schema = CollectionSchema(field_schemas, description="Auto-schema from CSV header")
    collection = Collection(args.collection, schema)

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

    print(f"Inserted {len(df)} rows into '{args.collection}'. Auto ID = {auto_id}")


if __name__ == "__main__":
    main()
