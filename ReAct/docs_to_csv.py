import csv
import os
import argparse
from typing import List, Dict, Any

DOCS: List[str] = [
    "Chroma is a lightweight open-source embedding database.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Hybrid search combines lexical (e.g., BM25) and semantic (dense) retrieval.",
    "Ubuntu 22.04 is a Long Term Support (LTS) release of Ubuntu.",
    "BM25 ranks documents based on term frequency and inverse document frequency.",
    "Vector databases store embeddings and provide approximate nearest neighbor search.",
    "FAISS supports CPU and GPU indexes; faiss-cpu works well for small demos.",
    "Chroma can persist collections locally and integrates easily in Python pipelines.",
    "OpenAI provides high-quality text embeddings like text-embedding-3-small.",
    "Dense + sparse fusion often improves retrieval over either method alone.",
]
DOC_IDS: List[str] = [f"doc-{i}" for i in range(len(DOCS))]

METADATA: List[Dict[str, Any]] = [
    {"topic": "db",      "source": "docs",  "year": 2024, "tags": ["chroma", "db"]},
    {"topic": "ann",     "source": "blog",  "year": 2023, "tags": ["faiss", "vectors"]},
    {"topic": "search",  "source": "blog",  "year": 2024, "tags": ["hybrid", "bm25", "dense"]},
    {"topic": "linux",   "source": "news",  "year": 2022, "tags": ["ubuntu", "lts"]},
    {"topic": "search",  "source": "paper", "year": 2019, "tags": ["bm25", "idf"]},
    {"topic": "db",      "source": "guide", "year": 2024, "tags": ["vector-db"]},
    {"topic": "ann",     "source": "docs",  "year": 2022, "tags": ["faiss", "cpu", "gpu"]},
    {"topic": "db",      "source": "docs",  "year": 2024, "tags": ["chroma", "python"]},
    {"topic": "embeds",  "source": "docs",  "year": 2025, "tags": ["openai", "embeddings"]},
    {"topic": "search",  "source": "blog",  "year": 2024, "tags": ["fusion"]},
]


def save_docs_to_csv(
    out_path: str,
    docs: List[str],
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    delimiter: str = ",",
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Determine all metadata keys present (union)
    meta_keys = set()
    for m in metadatas:
        meta_keys.update(m.keys())
    # fix ordering for predictable CSV columns
    ordered_meta_keys = ["topic", "source", "year", "tags"] + sorted(k for k in meta_keys if k not in {"topic", "source", "year", "tags"})

    fieldnames = ["id", "text"] + ordered_meta_keys

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for _id, doc, meta in zip(ids, docs, metadatas):
            row = {"id": _id, "text": doc}
            for k in ordered_meta_keys:
                v = meta.get(k)
                # serialize lists as semicolon-separated strings, leave scalars as-is
                if isinstance(v, list):
                    row[k] = ";".join(map(str, v))
                else:
                    row[k] = "" if v is None else str(v)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Write DOCS + METADATA to CSV.")
    parser.add_argument("--out", "-o", default="docs_with_metadata.csv", help="Output CSV path")
    parser.add_argument("--delimiter", "-d", default=",", help="CSV delimiter (default ',')")
    args = parser.parse_args()

    save_docs_to_csv(args.out, DOCS, DOC_IDS, METADATA, delimiter=args.delimiter)
    print(f"Wrote {len(DOCS)} rows to {args.out}")


if __name__ == "__main__":
    main()