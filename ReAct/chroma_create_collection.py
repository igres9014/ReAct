import os
import datetime
import shutil
import csv
import re                           # added import to fix NameError
from typing import Dict, List, Any

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidArgumentError
from openai import OpenAI

# Configuration
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

BASE_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_persist")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PERSIST_DIR = f"{BASE_PERSIST_DIR}_{TIMESTAMP}"
# Do NOT delete existing chroma_persist* folder by default.
# Set CHROMA_REMOVE_PERSIST=1 explicitly to enable removal.
REMOVE_PERSIST_IF_EXISTS = os.getenv("CHROMA_REMOVE_PERSIST", "0") == "1"


def embed_texts_openai(texts: List[str], model: str = OPENAI_MODEL) -> np.ndarray:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2-normalize for cosine/inner-product usage
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


def build_chroma_collection(
    docs: List[str],
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    embeddings: np.ndarray,
    collection_name: str,
    persist_dir: str = PERSIST_DIR,
    remove_persist: bool = REMOVE_PERSIST_IF_EXISTS,
) -> chromadb.Collection:
    # By default we will NOT remove an existing persistent dir to preserve data.
    # If remove_persist=True (CHROMA_REMOVE_PERSIST=1) is explicitly set, then remove.
    if remove_persist and os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"Warning: failed to remove {persist_dir}: {e}")

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))

    # If collection exists, delete it
    try:
        existing = client.list_collections()
        if collection_name in existing:
            client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection (use cosine space for normalized vectors)
    col = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    # Sanitize metadata (Chroma requires primitives)
    def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for k, v in meta.items():
            if isinstance(v, list):
                sanitized[k] = ", ".join(map(str, v))
            else:
                sanitized[k] = v
        return sanitized

    sanitized_metadatas = [sanitize_meta(m) for m in metadatas]

    # Try add, retry on embedding-dim mismatch by recreating collection
    try:
        col.add(ids=ids, documents=docs, metadatas=sanitized_metadatas, embeddings=embeddings.tolist())
    except InvalidArgumentError as e:
        # recreate collection and retry
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        col = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        col.add(ids=ids, documents=docs, metadatas=sanitized_metadatas, embeddings=embeddings.tolist())

    return col


if __name__ == "__main__":
    # Option: load documents from CSV or use default demo set
    use_csv = input("Load documents from CSV? (y/N): ").strip().lower() == "y"

    if use_csv:
        csv_path = input("Enter path to CSV file: ").strip()
        if not os.path.isfile(csv_path):
            raise RuntimeError(f"CSV file not found: {csv_path}")

        delim = input("Delimiter (default ','): ").strip() or ","
        # Let user optionally specify text/id columns; if empty, infer from header
        text_col = input("Text column name (leave empty to auto-detect): ").strip() or None
        id_col = input("ID column name (optional, leave empty to auto-generate or auto-detect 'id'): ").strip() or None
        meta_cols_input = input("Comma-separated metadata columns (optional, leave empty to use all other columns): ").strip()

        DOCS = []
        DOC_IDS = []
        METADATA = []

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            headers = reader.fieldnames or []
            if not headers:
                raise RuntimeError("CSV has no header row; please provide a header with column names.")

            # Auto-detect text_col if not provided: prefer common names, else first header
            if not text_col:
                for candidate in ("text", "content", "body", "document"):
                    if candidate in [h.lower() for h in headers]:
                        # pick the header matching candidate (preserve original case)
                        text_col = next(h for h in headers if h.lower() == candidate)
                        break
                else:
                    text_col = headers[0]

            # Auto-detect id_col if not provided and 'id' exists
            if not id_col:
                if any(h.lower() == "id" for h in headers):
                    id_col = next(h for h in headers if h.lower() == "id")
                else:
                    id_col = None

            # Determine metadata columns: either user-specified or all others except text_col and id_col
            if meta_cols_input:
                meta_cols = [c.strip() for c in meta_cols_input.split(",") if c.strip()]
            else:
                meta_cols = [h for h in headers if h not in {text_col, id_col}]

            # Validate text_col presence
            if text_col not in headers:
                raise RuntimeError(f"Detected text column '{text_col}' not found in CSV headers: {headers}")

            row_idx = 0
            for row in reader:
                text = (row.get(text_col) or "").strip()
                if not text:
                    # skip empty text rows
                    continue
                DOCS.append(text)
                if id_col and id_col in row and (row[id_col] or "").strip():
                    DOC_IDS.append(str(row[id_col].strip()))
                else:
                    DOC_IDS.append(f"doc-{row_idx}")

                md: Dict[str, Any] = {}
                for mc in meta_cols:
                    raw = row.get(mc, "")
                    if raw is None or raw == "":
                        md[mc] = None
                        continue
                    rv = raw.strip()
                    # try numeric conversion
                    try:
                        if rv.replace(".", "", 1).isdigit():
                            if "." in rv:
                                md[mc] = float(rv)
                            else:
                                md[mc] = int(rv)
                            continue
                    except Exception:
                        pass
                    # treat comma/semicolon-separated values as list
                    if "," in rv or ";" in rv:
                        parts = [p.strip() for p in re.split(r"[;,]", rv) if p.strip()]
                        md[mc] = parts
                    else:
                        md[mc] = rv
                METADATA.append(md)
                row_idx += 1

        if not DOCS:
            raise RuntimeError("No documents found in CSV after parsing.")

    else:
        # Demo usage: small document set (replace with your documents)
        DOCS = [
            "LangChain is a framework for developing applications powered by language models.",
            "Vector databases are essential for storing and retrieving embeddings efficiently.",
            "FAISS is a library for efficient similarity search of dense vectors.",
        ]
        DOC_IDS = [f"doc-{i}" for i in range(len(DOCS))]
        METADATA = [
            {"source": "docs", "category": "framework"},
            {"source": "tweet", "category": "database"},
            {"source": "docs", "category": "database"},
        ]

    print("Embedding documents with OpenAI:", OPENAI_MODEL)
    vectors = embed_texts_openai(DOCS)

    # --- User interaction: choose collection name and local persist folder ---
    default_coll = f"TRIAL_DATA_CHROMA_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    coll_input = input(f"Enter collection name [{default_coll}]: ").strip()
    coll_name = coll_input or default_coll

    default_persist = PERSIST_DIR
    persist_input = input(f"Enter local persist folder path [{default_persist}]: ").strip()
    persist_dir = os.path.expanduser(persist_input) if persist_input else default_persist
    persist_dir = os.path.abspath(persist_dir)

    # Ensure folder exists (will be created by build_chroma_collection if needed)
    if not os.path.exists(persist_dir):
        try:
            os.makedirs(persist_dir, exist_ok=True)
            print(f"Created persist dir: {persist_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to create persist dir '{persist_dir}': {e}")

    # Check for existing collection with same name in the chosen persist dir.
    # If found, warn the user and request a new name (loop until unique).
    try:
        client_check = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
        try:
            existing_cols = client_check.list_collections() or []
            # normalize to names if objects returned
            existing_names = []
            for c in existing_cols:
                if isinstance(c, str):
                    existing_names.append(c)
                else:
                    name = getattr(c, "name", None) or getattr(c, "id", None) or str(c)
                    existing_names.append(name)
        finally:
            try:
                client_check.close()
            except Exception:
                pass
    except Exception:
        existing_names = []

    while coll_name in existing_names:
        print(f"Warning: collection '{coll_name}' already exists in persist dir {persist_dir}.")
        suggestion = f"{coll_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_input = input(f"Enter a new collection name (or press Enter to use suggestion '{suggestion}'): ").strip()
        coll_name = new_input or suggestion
        # loop until coll_name not in existing_names

    # Optionally confirm removal behavior
    if REMOVE_PERSIST_IF_EXISTS:
        confirm = input(f"Configured to remove existing persist dir if present. Proceed with persist_dir={persist_dir}? (Y/n): ").strip().lower()
        if confirm == "n":
            print("Aborting per user request.")
            raise SystemExit(1)

    collection = build_chroma_collection(
        docs=DOCS,
        ids=DOC_IDS,
        metadatas=METADATA,
        embeddings=vectors,
        collection_name=coll_name,
        persist_dir=persist_dir,
        remove_persist=REMOVE_PERSIST_IF_EXISTS,
    )

    print(f"Created Chroma collection '{coll_name}' at: {os.path.abspath(persist_dir)}")