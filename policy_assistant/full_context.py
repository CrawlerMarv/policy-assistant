"""Approach 2: LLM without a vector index.

All 98 policies (~1.5k tokens) go into the prompt on every request; the LLM
picks the relevant one itself. `answer_closed_book` is an optional extra
baseline where the LLM gets no policy data at all.
"""
from __future__ import annotations

from .data import Result, load_policies
from .llm import ask_llm, closed_book_system, grounded_system


def answer(question: str) -> Result:
    return ask_llm("llm_full", grounded_system(load_policies()), question)


def answer_closed_book(question: str) -> Result:
    return ask_llm("llm_closed_book", closed_book_system(), question)
