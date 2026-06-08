"""
vector_store.py — ChromaDB persistence + local embedding helpers.

Embeddings are generated locally via sentence-transformers (no API key needed).
ChromaDB stores chunks in a persistent local directory (default: chroma_db/).
"""

import re
import chromadb
from chromadb.config import Settings

# sentence-transformers is imported lazily so import errors are surfaced
# only when the model is first used, not at module import time.
_embedding_model = None


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_embedding_model():
    """Return a cached SentenceTransformer instance."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using the local model.

    Returns a list of float lists (one per input text).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# ChromaDB client + collection helpers
# ---------------------------------------------------------------------------

def get_chroma_client(persist_directory: str = "chroma_db") -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False),
    )


def _safe_collection_name(project_id: str) -> str:
    """
    Convert an arbitrary project_id into a ChromaDB-safe collection name.

    ChromaDB requires: 3-63 chars, alphanumeric + underscores/hyphens,
    must start and end with alphanumeric.
    """
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", project_id)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_-")
    # Prepend prefix so the name always starts with a letter
    name = f"proj_{safe}"
    # Truncate to 63 chars
    return name[:63]


def get_collection(project_id: str, persist_directory: str = "chroma_db"):
    """
    Return (or create) the ChromaDB collection for this project.

    Uses cosine similarity (handled via normalized embeddings + dot-product
    under the hood when embeddings are L2-normalised).
    """
    client = get_chroma_client(persist_directory)
    collection_name = _safe_collection_name(project_id)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Core CRUD operations
# ---------------------------------------------------------------------------

def add_chunks(
    project_id: str,
    chunks: list[dict],
    persist_directory: str = "chroma_db",
) -> None:
    """
    Add (or upsert) a list of chunks into the project collection.

    Each chunk must have: chunk_id, text, metadata.
    """
    if not chunks:
        return

    collection = get_collection(project_id, persist_directory)

    ids        = [c["chunk_id"] for c in chunks]
    documents  = [c["text"]     for c in chunks]
    metadatas  = [_coerce_metadata(c["metadata"]) for c in chunks]
    embeddings = embed_texts(documents)

    # Upsert in batches of 500 to avoid memory pressure
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        collection.upsert(
            ids        = ids[start : start + batch_size],
            documents  = documents[start : start + batch_size],
            metadatas  = metadatas[start : start + batch_size],
            embeddings = embeddings[start : start + batch_size],
        )


def search_chunks(
    project_id: str,
    query: str,
    top_k: int = 8,
    chunk_type_filter: str | None = None,
    persist_directory: str = "chroma_db",
) -> list[dict]:
    """
    Semantic search over the project collection.

    Returns a list of result dicts:
        { "text": str, "metadata": dict, "distance": float, "score": float }

    If the collection is empty or the query finds nothing, returns [].
    """
    try:
        collection = get_collection(project_id, persist_directory)
        if collection.count() == 0:
            return []
    except Exception:
        return []

    query_embedding = embed_texts([query])[0]

    where_filter = None
    if chunk_type_filter:
        where_filter = {"chunk_type": {"$eq": chunk_type_filter}}

    kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        kwargs["where"] = where_filter

    try:
        results = collection.query(**kwargs)
    except Exception:
        return []

    output = []
    docs      = results.get("documents",  [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Cosine distance → similarity: score = 1 - distance (0–1 range, higher is better)
        score = max(0.0, 1.0 - dist)
        output.append({
            "text":     doc,
            "metadata": meta or {},
            "distance": dist,
            "score":    round(score, 4),
        })

    return output


def delete_project_collection(
    project_id: str,
    persist_directory: str = "chroma_db",
) -> None:
    """Delete the project's ChromaDB collection if it exists."""
    try:
        client = get_chroma_client(persist_directory)
        collection_name = _safe_collection_name(project_id)
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection may not exist; that's fine


def collection_count(
    project_id: str,
    persist_directory: str = "chroma_db",
) -> int:
    """Return the number of chunks in the project collection (0 if none)."""
    try:
        collection = get_collection(project_id, persist_directory)
        return collection.count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_metadata(meta: dict) -> dict:
    """
    ChromaDB requires metadata values to be str, int, float, or bool.
    Convert anything else to string.
    """
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean
