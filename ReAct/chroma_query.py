import os
import glob
import json
import datetime
import re
import math
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidArgumentError
from openai import OpenAI

BASE_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_persist")
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment")

def choose_persist_dir() -> str:
    candidates = sorted(glob.glob(f"{BASE_PERSIST_DIR}_*"), reverse=True)
    print("\nDetected chroma persist dirs:")
    for i, c in enumerate(candidates, 1):
        print(f" {i}) {c}")
    print(f" {len(candidates)+1}) Enter custom path")
    print(f" {len(candidates)+2}) Cancel / exit")

    while True:
        try:
            choice = int(input(f"Select persist dir (1-{len(candidates)+2}): ").strip())
        except ValueError:
            print("Invalid input")
            continue

        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        if choice == len(candidates) + 1:
            p = input("Enter full path to chroma persist dir: ").strip()
            if os.path.isdir(p):
                return p
            print("Path not found or not a directory")
        if choice == len(candidates) + 2:
            raise SystemExit("Cancelled by user")

def choose_collection(client: chromadb.PersistentClient) -> str:
    cols = client.list_collections()
    if not cols:
        raise RuntimeError("No collections found in selected persist dir")

    # Normalize to names (handle both str and Collection objects)
    names: List[str] = []
    for c in cols:
        if isinstance(c, str):
            names.append(c)
        else:
            # try common attributes, fallback to str()
            name = getattr(c, "name", None) or getattr(c, "id", None) or str(c)
            names.append(name)

    print("\nAvailable collections:")
    for i, name in enumerate(names, 1):
        print(f" {i}) {name}")

    while True:
        choice = input(f"Choose collection by number or name (1-{len(names)}): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return names[idx]
        elif choice in names:
            return choice
        print("Invalid choice")

def embed_query(text: str) -> List[float]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=OPENAI_MODEL, input=[text])
    return resp.data[0].embedding

def parse_where(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        print("Failed to parse filter JSON — ignoring filter")
        return None

def parse_where_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Simple filter language -> Chroma `where` dict.
    Syntax (clauses separated by ';'):
      field>=value
      field<=value
      field>value
      field<value
      field!=value
      field=value
      field in v1,v2,v3
    Example: "year>=2024; topic=search"
    """
    text = (text or "").strip()
    if not text:
        return None

    def to_number(s: str):
        s = s.strip()
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return s

    where: Dict[str, Any] = {}
    clauses = [c.strip() for c in re.split(r";\s*", text) if c.strip()]
    for clause in clauses:
        m = re.match(r"^(\w+)\s+in\s+(.+)$", clause, flags=re.I)
        if m:
            field = m.group(1)
            vals = [to_number(v) for v in re.split(r"\s*,\s*", m.group(2))]
            where[field] = {"$in": vals}
            continue

        m = re.match(r"^(\w+)\s*(>=|<=|!=|=|>|<)\s*(.+)$", clause)
        if m:
            field, op, raw = m.group(1), m.group(2), m.group(3)
            val = to_number(raw)
            if op == "=":
                where[field] = val
            elif op == "!=":
                where[field] = {"$ne": val}
            elif op == ">":
                where[field] = {"$gt": val}
            elif op == ">=":
                where[field] = {"$gte": val}
            elif op == "<":
                where[field] = {"$lt": val}
            elif op == "<=":
                where[field] = {"$lte": val}
            continue

        parts = re.split(r"\s*=\s*|\s+", clause, maxsplit=1)
        if len(parts) == 2:
            where[parts[0]] = to_number(parts[1])

    return where


def pretty_print(resp: Dict[str, Any]):
    """
    Handle Chroma query responses that may be nested (per-query lists).
    Print friendly output and show 'No results' when empty.
    """
    ids = resp.get("ids", [])
    # normalize nested lists for single-query responses
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        docs = resp.get("documents", [])
        docs = docs[0] if docs and isinstance(docs[0], list) else docs
        metas = resp.get("metadatas", [])
        metas = metas[0] if metas and isinstance(metas[0], list) else metas
        dists = resp.get("distances", [])
        dists = dists[0] if dists and isinstance(dists[0], list) else dists
    else:
        docs = resp.get("documents", [])
        metas = resp.get("metadatas", [])
        dists = resp.get("distances", [])

    if not ids:
        print("\nNo results.")
        return

    for i, _id in enumerate(ids):
        print(f"\n[{i+1}] id: {_id}")
        if metas and i < len(metas):
            print(" metadata:", metas[i])
        if docs and i < len(docs):
            print(" doc:", docs[i])
        if dists and i < len(dists):
            print(" distance:", dists[i])

def metadata_match(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """Simple evaluator for `where` produced by parse_where_text (supports =, $in, $gt/$gte/$lt/$lte)."""
    if not where:
        return True
    for k, cond in where.items():
        val = meta.get(k)
        if isinstance(cond, dict):
            if "$in" in cond:
                vals = cond["$in"]
                if isinstance(val, list):
                    if not any(x in val for x in vals):
                        return False
                else:
                    if val not in vals:
                        return False
            if "$gt" in cond:
                if not (isinstance(val, (int, float)) and val > cond["$gt"]):
                    return False
            if "$gte" in cond:
                if not (isinstance(val, (int, float)) and val >= cond["$gte"]):
                    return False
            if "$lt" in cond:
                if not (isinstance(val, (int, float)) and val < cond["$lt"]):
                    return False
            if "$lte" in cond:
                if not (isinstance(val, (int, float)) and val <= cond["$lte"]):
                    return False
            if "$ne" in cond:
                if val == cond["$ne"]:
                    return False
        else:
            if val != cond:
                return False
    return True

def _normalize_scores(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        return {k: 1.0 for k in d.keys()}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

def keyword_search_local(collection: chromadb.api.models.Collection.Collection, query: str, where: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Simple keyword scoring over all documents in the collection.
    Score = token overlap count (can be 0). Returns dict id->score.
    """
    try:
        all_docs = collection.get(include=["ids", "documents", "metadatas"])
    except Exception:
        # fallback: try without include (older API)
        all_docs = collection.get()
    ids = all_docs.get("ids", [])
    docs = all_docs.get("documents", [])
    metas = all_docs.get("metadatas", [])

    # normalize nested single-query shapes
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if docs and isinstance(docs[0], list):
        docs = docs[0]
    if metas and isinstance(metas[0], list):
        metas = metas[0]

    q_tokens = [t for t in re.split(r"\W+", query.lower()) if t]
    scores: Dict[str, float] = {}
    for _id, doc, meta in zip(ids, docs, metas):
        if where and not metadata_match(meta or {}, where):
            continue
        doc_tokens = [t for t in re.split(r"\W+", (doc or "").lower()) if t]
        if not doc_tokens:
            continue
        overlap = sum(1 for t in set(q_tokens) if t in doc_tokens)
        if overlap > 0:
            scores[_id] = float(overlap)
    return scores

def _extract_semantic_scores(resp: Dict[str, Any]) -> Dict[str, float]:
    """Extract mapping id->score from Chroma query response (handles nested lists)."""
    ids = resp.get("ids", [])
    dists = resp.get("distances", [])
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if dists and isinstance(dists[0], list):
        dists = dists[0]
    if not ids:
        return {}
    # Chroma distances may be similarity scores; keep as-is
    return {str(_id): float(d) if (d is not None) else 0.0 for _id, d in zip(ids, dists)}

def main():
    persist_dir = choose_persist_dir()
    print("Using persist dir:", persist_dir)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
    try:
        coll_name = choose_collection(client)
        print("Selected collection:", coll_name)
        collection = client.get_collection(name=coll_name)

        while True:
            # New: embed locally with OpenAI and use embedding-query
            print("\nQuery modes:")
            print(" 1) Embedding query (use OpenAI to embed locally and query by embedding)")
            print(" 2) Exit")
            mode = input("Choose mode (1-2): ").strip()
            if mode == "2":
                break
            if mode != "1":
                print("Invalid mode")
                continue

            qtext = input("Enter query text (will be embedded with OpenAI): ").strip()
            if not qtext:
                print("Empty query, try again")
                continue

            print("\nFilter examples:")
            print("  year>=2024; topic=search")
            print("  tags in chroma,db; source=docs")
            flt_text = input("Optional filter (e.g. \"year>=2024; topic=search\") or press Enter for none: ").strip()
            where = parse_where_text(flt_text)

            try:
                top_k = int(input("Top-k (default 5): ").strip() or "5")
            except ValueError:
                top_k = 5

            # NEW: ask whether to include keyword search and mixing weight
            include_kw = input("Include keyword matching in results? (y/N): ").strip().lower() == "y"
            if include_kw:
                try:
                    kw_weight = float(input("Keyword weight (0.0 - 1.0, default 0.3): ").strip() or "0.3")
                except ValueError:
                    kw_weight = 0.3
                kw_weight = max(0.0, min(1.0, kw_weight))
                sem_weight = 1.0 - kw_weight
            else:
                kw_weight = 0.0
                sem_weight = 1.0

            try:
                # Embed locally with OpenAI and query using embeddings
                vec = embed_query(qtext)
                resp = collection.query(query_embeddings=[vec], n_results=top_k, where=where)
                sem_scores = _extract_semantic_scores(resp)

                if include_kw:
                    kw_scores = keyword_search_local(collection, qtext, where=where)
                    sem_norm = _normalize_scores(sem_scores)
                    kw_norm = _normalize_scores(kw_scores)
                    # combine keys
                    all_ids = set(list(sem_norm.keys()) + list(kw_norm.keys()))
                    fused_scores = {}
                    for _id in all_ids:
                        s = sem_norm.get(_id, 0.0) * sem_weight
                        k = kw_norm.get(_id, 0.0) * kw_weight
                        fused_scores[_id] = s + k
                    # pick top-k ids
                    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
                    result_ids = [rid for rid, _ in ranked]
                    # fetch docs/metas for printing
                    fetch = collection.get(ids=result_ids, include=["documents", "metadatas"])
                    # normalize nested shapes
                    ids_fetch = fetch.get("ids", [])
                    docs_fetch = fetch.get("documents", [])
                    metas_fetch = fetch.get("metadatas", [])
                    if ids_fetch and isinstance(ids_fetch[0], list):
                        ids_fetch = ids_fetch[0]
                        docs_fetch = docs_fetch[0] if docs_fetch and isinstance(docs_fetch[0], list) else docs_fetch
                        metas_fetch = metas_fetch[0] if metas_fetch and isinstance(metas_fetch[0], list) else metas_fetch
                    print("\nFused results (semantic + keyword):")
                    for i, rid in enumerate(result_ids):
                        doc = docs_fetch[i] if i < len(docs_fetch) else ""
                        meta = metas_fetch[i] if i < len(metas_fetch) else {}
                        score = fused_scores.get(rid, 0.0)
                        print(f"\n[{i+1}] id: {rid}  score: {score:.4f}")
                        print(" metadata:", meta)
                        print(" doc:", doc)
                else:
                    print("\nQuery results:")
                    pretty_print(resp)
            except InvalidArgumentError as e:
                print("Chroma InvalidArgumentError:", e)
                print("Possible embedding-dimension mismatch between collection and model.")
            except Exception as e:
                print("Query failed:", e)

            cont = input("\nRun another query? (y/N): ").strip().lower()
            if cont != "y":
                break
    finally:
        try:
            client.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()