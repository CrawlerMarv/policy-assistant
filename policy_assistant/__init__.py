"""Company policy assistant – one entry point for website, Slack bot and evaluation.

    from policy_assistant import ask
    result = ask("llm_rag", "How many vacation days do I get?")
"""
from __future__ import annotations

import time
from typing import Callable

try:  # load ANTHROPIC_API_KEY etc. from .env when running locally
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import full_context, rag, rules
from .data import Policy, Result, load_policies, policies_by_title

APPROACHES: dict[str, Callable[[str], Result]] = {
    "rules": rules.answer,
    "llm_full": full_context.answer,
    "llm_rag": rag.answer,
    "llm_closed_book": full_context.answer_closed_book,  # optional extra baseline
}

LABELS = {
    "rules": "Rules-based search (BM25)",
    "llm_full": "LLM without vector index",
    "llm_rag": "LLM with vector index (RAG)",
    "llm_closed_book": "LLM without policy data",
}

# The three approaches the assignment requires
DEFAULT_APPROACHES = ["rules", "llm_full", "llm_rag"]


def warm_up(approaches=DEFAULT_APPROACHES) -> None:
    """Build indexes up front so one-off setup cost isn't counted as response time."""
    rules.get_index()
    if "llm_rag" in approaches:
        rag.get_index()


def ask(approach: str, question: str) -> Result:
    start = time.perf_counter()
    result = APPROACHES[approach](question)
    result.seconds = max(0.0, time.perf_counter() - start - result.retry_wait)
    return result


__all__ = ["ask", "warm_up", "APPROACHES", "LABELS", "DEFAULT_APPROACHES",
           "Policy", "Result", "load_policies", "policies_by_title"]
