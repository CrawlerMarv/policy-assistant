"""Approach 3: LLM with a vector index (retrieval-augmented generation).

Policies are embedded once with a small local embedding model (fastembed,
BAAI/bge-small-en-v1.5, runs on CPU, no API key). Per question, the top-k most
similar policies are retrieved and only those are passed to the LLM.

The index is a flat cosine-similarity index in NumPy – for 98 documents that is
exactly what FAISS IndexFlatIP would do, without the extra dependency.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .data import Policy, Result, load_policies
from .llm import ask_llm, grounded_system

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))


class VectorIndex:
    def __init__(self, policies: tuple[Policy, ...]):
        from fastembed import TextEmbedding  # model (~70 MB) is downloaded on first use

        self.policies = policies
        self.model = TextEmbedding(EMBED_MODEL)
        docs = [f"{p.title}: {p.text}" for p in policies]
        vecs = np.array(list(self.model.passage_embed(docs)), dtype=np.float32)
        self.matrix = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Policy, float]]:
        q = np.array(next(iter(self.model.query_embed(query))), dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.policies[i], float(scores[i])) for i in top]


@lru_cache(maxsize=1)
def get_index() -> VectorIndex:
    """Built once per process. Call at startup so index building isn't counted as response time."""
    return VectorIndex(load_policies())


def answer(question: str) -> Result:
    hits = get_index().search(question)
    retrieved = [f"{p.title} ({s:.2f})" for p, s in hits]
    return ask_llm("llm_rag", grounded_system([p for p, _ in hits]), question, retrieved)
