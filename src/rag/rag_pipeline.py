"""
rag_pipeline.py — High-level RAG orchestration functions.

Three public entry points:
    build_project_index  — chunk all DataFrames and store in ChromaDB
    ask_project_question — retrieve + Claude answer for a free-form question
    explain_match        — targeted explanation for a specific match result
"""

from .chunk_builder import build_all_chunks, dataframe_has_rows
from .vector_store import (
    add_chunks,
    delete_project_collection,
    collection_count,
)
from .retriever import retrieve_context
from .claude_answerer import answer_with_claude


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def build_project_index(
    project_id: str,
    pdf_pages_df=None,
    pdf_records_df=None,
    csv_df=None,
    matches_df=None,
    clusters_df=None,
    reset_existing: bool = True,
    persist_directory: str = "chroma_db",
) -> dict:
    """
    Build (or rebuild) the vector index for a project.

    Parameters
    ----------
    project_id      : namespace key for the ChromaDB collection
    *_df            : DataFrames to index (any can be None)
    reset_existing  : if True, wipe the existing collection first
    persist_directory : ChromaDB storage path

    Returns
    -------
    {
        "project_id":        str,
        "chunks_indexed":    int,
        "chunk_type_counts": { chunk_type: count, ... }
    }
    """
    if reset_existing:
        delete_project_collection(project_id, persist_directory)

    chunks = build_all_chunks(
        project_id=project_id,
        pdf_pages_df=pdf_pages_df,
        pdf_records_df=pdf_records_df,
        csv_df=csv_df,
        matches_df=matches_df,
        clusters_df=clusters_df,
    )

    add_chunks(project_id, chunks, persist_directory)

    # Count by type
    type_counts: dict[str, int] = {}
    for chunk in chunks:
        ct = chunk.get("chunk_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    return {
        "project_id":        project_id,
        "chunks_indexed":    len(chunks),
        "chunk_type_counts": type_counts,
    }


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------

def ask_project_question(
    project_id: str,
    question: str,
    top_k: int = 8,
    chunk_type_filter: str | None = None,
    persist_directory: str = "chroma_db",
) -> dict:
    """
    Retrieve evidence and generate a grounded Claude answer.

    Returns
    -------
    {
        "answer":         str   — Claude's answer
        "context_chunks": list  — retrieved chunk dicts
    }
    """
    context_chunks = retrieve_context(
        project_id=project_id,
        question=question,
        top_k=top_k,
        chunk_type_filter=chunk_type_filter,
        persist_directory=persist_directory,
    )

    answer = answer_with_claude(question=question, context_chunks=context_chunks)

    return {
        "answer":         answer,
        "context_chunks": context_chunks,
    }


# ---------------------------------------------------------------------------
# Match explanation
# ---------------------------------------------------------------------------

def explain_match(
    project_id: str,
    match_id=None,
    match_row=None,
    persist_directory: str = "chroma_db",
) -> dict:
    """
    Generate a targeted explanation for a specific match result.

    Parameters
    ----------
    project_id : project namespace
    match_id   : the match_id value (str or int) to look up
    match_row  : optional — a single-row dict/Series with match fields
                 (used to build a richer question when provided)

    Returns
    -------
    { "answer": str, "context_chunks": list }
    """
    if match_row is not None:
        # Build a detailed question from the row's fields
        def _g(field):
            try:
                val = match_row.get(field, "") if hasattr(match_row, "get") else ""
                return str(val).strip() if val else ""
            except Exception:
                return ""

        mid       = _g("match_id") or str(match_id or "unknown")
        conf      = _g("confidence")
        score     = _g("match_score")
        reason    = _g("match_reason")
        pdf_carr  = _g("pdf_carrier_name")
        pdf_addr  = _g("pdf_carrier_address")
        csv_carr  = _g("csv_carrier_name")
        csv_addr  = _g("csv_carrier_address")
        notes     = _g("review_notes")

        question = (
            f"Explain why match ID {mid} was flagged as a possible investigative lead. "
            f"Confidence: {conf}. Score: {score}. "
            f"PDF carrier: {pdf_carr} at {pdf_addr}. "
            f"CSV carrier: {csv_carr} at {csv_addr}. "
            f"Reason: {reason}. Review notes: {notes}. "
            f"Include address similarity, different carrier/DOT characteristics, "
            f"source evidence from the index, and caveats."
        )
    elif match_id is not None:
        question = (
            f"Explain why match ID {match_id} was flagged. "
            f"Include address similarity, different carrier/DOT characteristics, "
            f"source evidence, and caveats."
        )
    else:
        question = (
            "Explain why the top match result was flagged as a possible "
            "investigative lead. Include address similarity, carrier/DOT "
            "differences, source evidence, and caveats."
        )

    return ask_project_question(
        project_id=project_id,
        question=question,
        top_k=10,
        persist_directory=persist_directory,
    )
