"""Runs every test question through each approach and writes
results/results.csv (one row per question x approach) and results/summary.csv.

    python evaluate.py                       # the three required approaches
    python evaluate.py -a rules              # only one approach (no API key needed)
    python evaluate.py --fresh               # ignore earlier results and ask everything again

Resumable: answers already in results/results.csv for the same model are reused, so
after hitting a daily quota you simply run the script again the next day and only the
missing questions are sent to the API. With LLM_PROVIDER=gemini it pauses 7 s between
LLM calls (free-tier minute limit); pauses and retry waits are not part of the measured time.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import time
from pathlib import Path

from policy_assistant import APPROACHES, DEFAULT_APPROACHES, LABELS, Result, ask, llm, policies_by_title, warm_up

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "data" / "test_questions.csv"
OUT = ROOT / "results"
RESULTS = OUT / "results.csv"

NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def numbers(text: str) -> set[str]:
    text = text.lower().replace(",", "")
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def unsupported_reason(r: Result, expected: str | None) -> str:
    """Empty string if the answer is backed by the policy database, otherwise the reason.

    Checks: (1) cited policy doesn't exist, (2) the question isn't covered by any
    policy but the approach still answered, (3) the answer contains numbers that
    appear neither in the cited policy nor in the question.
    """
    if r.policy_title is None:
        return ""  # approach declined – nothing claimed
    policy = policies_by_title().get(r.policy_title)
    if policy is None:
        return f"cited policy not in database: {r.policy_title}"
    if expected is None:
        return "answered a question no policy covers"
    extra = numbers(r.answer) - numbers(policy.text) - numbers(r.question)
    if extra:
        return f"numbers not in policy: {', '.join(sorted(extra))}"
    return ""


def model_tag(approach: str) -> str:
    return "—" if approach == "rules" else f"{llm.PROVIDER}/{llm.MODEL}"


def load_previous(fresh: bool) -> dict[tuple[str, str], dict]:
    """Successful LLM rows from an earlier run with the same model (rules is always re-run)."""
    if fresh or not RESULTS.exists():
        return {}
    keep = {}
    with RESULTS.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["approach"] == "rules" or r.get("error") or r.get("model") != model_tag(r["approach"]):
                continue
            r["correct"] = r["correct"] == "True"
            r["unsupported"] = r["unsupported"] == "True"
            r["seconds"] = float(r["seconds"])
            r["total_tokens"] = int(r["total_tokens"])
            keep[(r["approach"], r["question"])] = r
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--approaches", nargs="+", default=DEFAULT_APPROACHES, choices=list(APPROACHES))
    parser.add_argument("--pause", type=float, default=7.0 if llm.PROVIDER == "gemini" else 0.0,
                        help="seconds to wait between LLM calls")
    parser.add_argument("--fresh", action="store_true", help="ignore earlier results")
    args = parser.parse_args()
    print(f"LLM: {llm.PROVIDER} / {llm.MODEL}, pause {args.pause:.0f}s between LLM calls")

    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        tests = list(csv.DictReader(f))

    previous = load_previous(args.fresh)
    if previous:
        print(f"Reusing {len(previous)} answers from the previous run (use --fresh to redo them)")
    print()

    warm_up(args.approaches)
    if any(a != "rules" for a in args.approaches):
        llm.warm_up_connection()

    rows, quota_hit = [], False
    for t in tests:
        expected = t["expected_policy"] or None
        for name in args.approaches:
            key = (name, t["question"])
            if key in previous:
                rows.append(previous[key])
                continue
            base = {"approach": name, "question": t["question"], "type": t["type"],
                    "expected_policy": expected or "", "model": model_tag(name)}
            if name != "rules" and quota_hit:
                rows.append({**base, "error": "skipped: quota used up"})
                continue
            try:
                r = ask(name, t["question"])
            except llm.QuotaExceeded as e:
                quota_hit = True
                print(f"\n*** {e}. Remaining LLM questions are skipped – run the script again "
                      f"after the quota resets (midnight Pacific time).\n")
                rows.append({**base, "error": str(e)})
                continue
            except Exception as e:  # e.g. API still overloaded after all retries
                print(f"[{name:16}] ERROR {t['question'][:55]}: {str(e)[:120]}")
                rows.append({**base, "error": str(e)[:300]})
                continue
            finally:
                if name != "rules" and not quota_hit:
                    time.sleep(args.pause)
            reason = unsupported_reason(r, expected)
            rows.append({**r.to_dict(), **base, "correct": r.policy_title == expected,
                         "unsupported": bool(reason), "unsupported_reason": reason, "error": ""})
            print(f"[{name:16}] {'✓' if r.policy_title == expected else '✗'} "
                  f"{'!' if reason else ' '} {t['question'][:55]:55} -> {r.policy_title or '—'}")

    OUT.mkdir(exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with RESULTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    summary = []
    for name in args.approaches:
        failed = [r for r in rows if r["approach"] == name and r.get("error")]
        rs = [r for r in rows if r["approach"] == name and not r.get("error")]
        if not rs:
            print(f"{LABELS[name]}: no answers yet")
            continue
        in_scope = [r for r in rs if r["type"] != "out_of_scope"]
        out_scope = [r for r in rs if r["type"] == "out_of_scope"]
        summary.append({
            "approach": LABELS[name],
            "model": model_tag(name),
            "answered": f"{len(rs)}/{len(tests)}",
            "accuracy": f"{sum(r['correct'] for r in rs) / len(rs):.0%}",
            "correct_policy_in_scope": f"{sum(r['correct'] for r in in_scope)}/{len(in_scope)}",
            "correctly_declined_out_of_scope": f"{sum(r['correct'] for r in out_scope)}/{len(out_scope)}",
            "unsupported_rate": f"{sum(r['unsupported'] for r in rs) / len(rs):.0%}",
            "avg_seconds": f"{statistics.mean(r['seconds'] for r in rs):.2f}",
            "median_seconds": f"{statistics.median(r['seconds'] for r in rs):.2f}",
            "avg_tokens": f"{statistics.mean(r['total_tokens'] for r in rs):.0f}",
        })
    if not summary:
        return
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print("\n| " + " | ".join(summary[0]) + " |")
    print("|" + "---|" * len(summary[0]))
    for s in summary:
        print("| " + " | ".join(s.values()) + " |")
    missing = sum(1 for r in rows if r.get("error"))
    if missing:
        print(f"\n{missing} answers missing – run `python evaluate.py` again later; "
              f"existing answers are kept.")


if __name__ == "__main__":
    main()
