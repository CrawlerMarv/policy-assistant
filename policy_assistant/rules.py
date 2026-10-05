"""Approach 1: rules-based keyword search (BM25), no LLM.

Returns the text of the best-matching policy verbatim, so it can never
invent content, but it only finds policies that share words with the question.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import snowballstemmer

from .data import Policy, Result, load_policies

# Minimum BM25 score before we accept a hit. Below this we answer "no policy found".
# Chosen from the scores in results/results.csv (column "retrieved"): real hits score ~8+, noise ~3-5.
MIN_SCORE = 5.5
K1, B = 1.5, 0.75
TITLE_WEIGHT = 2  # title words count twice

STOPWORDS = set("""
a an the and or but if of to in on at for from by with about as into over under
is are was were be been being am do does did done have has had having can could
may might must shall should will would i me my we our you your he she it they them
their this that these those what which who whom when where why how there here
not no yes any all some each every per than then so too very just also only
get got getting company policy policies employee employees allowed allow
""".split())

# Hand-written rules. Every synonym you add is manual work – that's the trade-off
# of this approach. Example: {"wfh": "remote", "jeans": "dress"}
SYNONYMS: dict[str, str] = {}

_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [SYNONYMS.get(w, w) for w in words]
    return _stemmer.stemWords([w for w in words if w not in STOPWORDS])


class BM25Index:
    def __init__(self, policies: tuple[Policy, ...]):
        self.policies = policies
        self.docs = [tokenize(p.title) * TITLE_WEIGHT + tokenize(p.text) for p in policies]
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        n = len(self.docs)
        df = Counter(term for d in self.docs for term in set(d))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Policy, float]]:
        terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * self.doc_len[i] / self.avg_len))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.policies[i], scores[i]) for i in order]


@lru_cache(maxsize=1)
def get_index() -> BM25Index:
    return BM25Index(load_policies())


def answer(question: str) -> Result:
    hits = get_index().search(question, k=3)
    best, score = hits[0]
    retrieved = [f"{p.title} ({s:.2f})" for p, s in hits]
    if score < MIN_SCORE:
        return Result("rules", question, "I couldn't find a policy that matches your question.", None,
                      retrieved=retrieved)
    return Result("rules", question, best.text, best.title, retrieved=retrieved)
