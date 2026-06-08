"""
claude_answerer.py — Generate source-grounded answers using Claude.

The Claude API key is read from the ANTHROPIC_API_KEY environment variable
(or from a .env file via python-dotenv). It is never hardcoded.
"""

import os
from .retriever import format_context_for_claude

# Change this constant to switch models without touching other code.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are an investigative document assistant for a document-matching system.

Use only the provided context chunks to answer the question.
Do not invent facts or reference information not present in the context.
If the context is insufficient to answer the question, clearly state that \
the evidence is insufficient.

These results are investigative leads, not proof of wrongdoing.
Never state that a carrier is definitely a chameleon carrier or is committing fraud.
Use terms such as:
  - possible lead
  - possible related carrier
  - address-based match
  - requires manual review
  - not proof of wrongdoing
  - source-backed investigative lead

When available, cite source references (PDF page number, CSV row ID, match ID, \
address cluster, chunk ID) to show where the evidence comes from.
Clearly separate direct evidence from inference or interpretation.
Be concise but complete.

Structure your response as:
1. Direct answer
2. Supporting evidence
3. Source references
4. Caveats / uncertainty
"""


def _load_api_key() -> str | None:
    """Load ANTHROPIC_API_KEY from environment (with .env fallback)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass  # python-dotenv not installed; rely on system environment
    return os.environ.get("ANTHROPIC_API_KEY", "").strip() or None


def answer_with_claude(
    question: str,
    context_chunks: list[dict],
    model: str | None = None,
) -> str:
    """
    Send the question + retrieved context to Claude and return a grounded answer.

    Parameters
    ----------
    question        : the user's natural-language question
    context_chunks  : list of chunk dicts from retriever.retrieve_context()
    model           : Claude model string; defaults to DEFAULT_CLAUDE_MODEL

    Returns
    -------
    A string answer.  On error, returns a human-readable error message
    (does not raise).
    """
    if not context_chunks:
        return (
            "No relevant evidence was found in the indexed project documents.\n\n"
            "Try building the RAG index first (click 'Build / Refresh RAG Index'), "
            "or rephrase your question."
        )

    api_key = _load_api_key()
    if not api_key:
        return (
            "ANTHROPIC_API_KEY is not set.\n\n"
            "Add it to a .env file in the project root:\n"
            "    ANTHROPIC_API_KEY=your_key_here\n\n"
            "Then restart the app."
        )

    try:
        import anthropic
    except ImportError:
        return (
            "The 'anthropic' package is not installed.\n"
            "Run: pip install anthropic"
        )

    context_text = format_context_for_claude(context_chunks)

    user_message = (
        f"Question: {question}\n\n"
        f"Context from indexed project documents:\n"
        f"{context_text}"
    )

    chosen_model = model or DEFAULT_CLAUDE_MODEL

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=chosen_model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return (
            "Authentication failed: the ANTHROPIC_API_KEY appears to be invalid.\n"
            "Check that the key in your .env file is correct."
        )
    except anthropic.RateLimitError:
        return "Rate limit reached. Please wait a moment and try again."
    except Exception as exc:
        return f"Claude API error: {exc}"
