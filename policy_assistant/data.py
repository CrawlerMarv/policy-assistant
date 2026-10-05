"""Loading the policy database and the shared result object."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "company_policies.csv"


@dataclass(frozen=True)
class Policy:
    title: str
    department: str
    text: str
    category: str

    def as_context(self) -> str:
        """One line per policy, used inside LLM prompts."""
        return f"[{self.title}] ({self.category}) {self.text}"


@lru_cache(maxsize=1)
def load_policies() -> tuple[Policy, ...]:
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return tuple(
            Policy(
                title=row["title"].strip(),
                department=row["department"].strip(),
                text=row["policy_text"].strip(),
                category=row["category"].strip(),
            )
            for row in csv.DictReader(f)
        )


@lru_cache(maxsize=1)
def policies_by_title() -> dict[str, Policy]:
    return {p.title: p for p in load_policies()}


def match_title(raw: str | None) -> str | None:
    """Map a title returned by an LLM onto an exact database title.

    Returns the raw string unchanged if nothing matches, so fabricated
    titles stay visible and can be flagged during evaluation.
    """
    if not raw:
        return None
    raw = raw.strip().strip("[]").strip()
    lookup = {t.lower(): t for t in policies_by_title()}
    return lookup.get(raw.lower(), raw)


@dataclass
class Result:
    approach: str
    question: str
    answer: str
    policy_title: str | None  # None = approach says no policy covers the question
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved: list[str] = field(default_factory=list)  # candidates shown to the LLM (RAG) or top hits (rules)
    retry_wait: float = 0.0  # seconds spent waiting before retries (excluded from `seconds`)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def policy_text(self) -> str | None:
        p = policies_by_title().get(self.policy_title or "")
        return p.text if p else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["policy_text"] = self.policy_text
        d["retrieved"] = "; ".join(self.retrieved)
        return d
