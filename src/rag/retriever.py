"""
retriever.py — Retrieve relevant chunks from ChromaDB and format for Claude.
"""

from .vector_store import search_chunks

# Max characters to show per chunk in the formatted context string
_CHUNK_TEXT_LIMIT = 2500


def retrieve_context(
    project_id: str,
    question: str,
    top_k: int = 8,
    chunk_type_filter: str | None = None,
    persist_directory: str = "chroma_db",
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for *question* from the project index.

    Parameters
    ----------
    project_id         : project namespace in ChromaDB
    question           : the user's natural-language query
    top_k              : number of results to return
    chunk_type_filter  : restrict to one chunk type (e.g. "match_result")
    persist_directory  : ChromaDB storage path

    Returns
    -------
    List of chunk dicts: { text, metadata, distance, score }
    """
    return search_chunks(
        project_id=project_id,
        query=question,
        top_k=top_k,
        chunk_type_filter=chunk_type_filter,
        persist_directory=persist_directory,
    )


def format_context_for_claude(context_chunks: list[dict]) -> str:
    """
    Format a list of retrieved chunks into a readable context block for Claude.

    Each chunk is labelled with its index, type, key metadata fields, and
    a (possibly truncated) text excerpt.
    """
    if not context_chunks:
        return "(No relevant context chunks retrieved.)"

    parts = []
    for idx, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        text = chunk.get("text", "")

        # Truncate long texts
        if len(text) > _CHUNK_TEXT_LIMIT:
            text = text[:_CHUNK_TEXT_LIMIT] + "\n[...truncated]"

        # Build a compact metadata summary
        meta_lines = []
        for key in (
            "chunk_type", "match_id", "source_file", "page_number",
            "csv_row_id", "row_id", "pdf_record_id",
            "carrier_name", "address", "usdot", "vin",
            "confidence", "match_score", "match_reason",
        ):
            val = meta.get(key, "")
            if val:
                meta_lines.append(f"{key}: {val}")

        score = chunk.get("score", "")
        if score != "":
            meta_lines.append(f"similarity_score: {score}")

        meta_block = "\n".join(meta_lines) if meta_lines else "(no metadata)"

        parts.append(
            f"CONTEXT CHUNK {idx}\n"
            f"{meta_block}\n"
            f"text:\n{text}"
        )

    return "\n\n" + "\n\n".join(parts)
