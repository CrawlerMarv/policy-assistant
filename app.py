"""Comparison website:  streamlit run app.py

Test questions show the stored results from evaluate.py (no API quota used);
your own questions are answered live by all three approaches.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Company Policy Assistant", page_icon="📘", layout="wide")

from evaluate import numbers  # noqa: E402
from policy_assistant import DEFAULT_APPROACHES, LABELS, ask, llm, policies_by_title, warm_up  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "results.csv"
SUMMARY = ROOT / "results" / "summary.csv"
QUESTIONS = ROOT / "data" / "test_questions.csv"
COMPARISON = ROOT / "comparison.md"
OWN_QUESTION = "Write your own question…"


@st.cache_resource(show_spinner="Building the search indexes…")
def init() -> bool:
    warm_up(DEFAULT_APPROACHES)
    return True


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, keep_default_na=False) if path.exists() else None


@st.cache_data(show_spinner=False, max_entries=500)
def live_answer(approach: str, question: str) -> dict:
    """Cached, so asking the same question twice doesn't use API quota again."""
    return ask(approach, question).to_dict()


def grounding_check(d: dict) -> tuple[str, str]:
    """Same checks as evaluate.py, minus the one that needs to know the right answer."""
    title = d.get("policy_title") or None
    if not title:
        return "neutral", "No policy cited: the approach says the database doesn't cover this."
    policy = policies_by_title().get(title)
    if policy is None:
        return "warn", f"Cites “{title}”, which is not in the policy database."
    extra = numbers(d["answer"]) - numbers(policy.text) - numbers(d["question"])
    if extra:
        return "warn", f"Mentions {', '.join(sorted(extra))}, which the cited policy doesn't state."
    return "ok", "The cited policy exists and the answer adds no figures beyond it."


def fmt_time(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms" if seconds < 0.1 else f"{seconds:.2f} s"


def show_answer(col, approach: str, d: dict | None, error: str | None) -> None:
    with col:
        st.subheader(LABELS[approach])
        if error:
            st.warning(error)
            return
        st.write(d["answer"])
        title = d.get("policy_title") or None
        st.markdown(f"**Relevant policy:** {title or 'none found'}")
        if title and d.get("policy_text"):
            st.caption(f"Policy text: {d['policy_text']}")
        m1, m2 = st.columns(2)
        m1.metric("Response time", fmt_time(float(d["seconds"])))
        m2.metric("Tokens", f"{int(d['total_tokens']):,}")
        level, message = grounding_check(d)
        {"ok": st.success, "warn": st.warning, "neutral": st.info}[level](message)
        if d.get("retrieved"):
            st.caption(f"Candidates considered: {d['retrieved']}")


init()
results = load_csv(RESULTS)
summary = load_csv(SUMMARY)
tests = load_csv(QUESTIONS)

st.title("Company Policy Assistant")
st.write(
    "Ask a question about company policy and see how three approaches answer it from the same "
    "database of 98 policies: a rules-based keyword search, a language model that reads the whole "
    "database, and a language model that first retrieves the most relevant policies from a vector index."
)

# --- Ask -------------------------------------------------------------------------------------
st.header("Compare the answers")
options = [OWN_QUESTION] + (tests["question"].tolist() if tests is not None else [])
choice = st.selectbox("Pick one of the 20 test questions or write your own", options, index=1 if len(options) > 1 else 0)
own = st.text_input("Your question", placeholder="e.g. Can I take my work laptop on holiday?") if choice == OWN_QUESTION else ""
question = (own if choice == OWN_QUESTION else choice).strip()

stored = None
if choice != OWN_QUESTION and results is not None:
    stored = results[(results["question"] == question) & (results["error"] == "")]

if question and stored is not None and not stored.empty:
    st.caption("Stored result from the evaluation run (no API call). Write your own question to see live answers.")
    cols = st.columns(len(DEFAULT_APPROACHES))
    for col, approach in zip(cols, DEFAULT_APPROACHES):
        row = stored[stored["approach"] == approach]
        show_answer(col, approach, row.iloc[0].to_dict() if not row.empty else None,
                    None if not row.empty else "No stored result for this approach.")
elif question:
    if st.button("Compare answers", type="primary") or st.session_state.get("last_q") == question:
        st.session_state["last_q"] = question
        cols = st.columns(len(DEFAULT_APPROACHES))
        for col, approach in zip(cols, DEFAULT_APPROACHES):
            d, error = None, None
            with st.spinner(f"Asking {LABELS[approach].lower()}…"):
                try:
                    d = live_answer(approach, question)
                except llm.QuotaExceeded:
                    error = ("Today's free API quota is used up. The stored results still work; "
                             "live questions are available again after midnight Pacific time.")
                except Exception as e:
                    error = f"The model could not be reached: {str(e)[:200]}"
            show_answer(col, approach, d, error)
        st.caption(f"Language model: {llm.PROVIDER} / {llm.MODEL}. Repeated questions are answered from cache.")

# --- Evaluation ------------------------------------------------------------------------------
st.header("Results on 20 test questions")
if summary is None or results is None:
    st.info("No evaluation results yet. Run `python evaluate.py` and commit the results folder.")
else:
    st.write(
        "Seven questions use the policy's own wording, seven are paraphrased, one asks for a detail "
        "the matching policy doesn't state, and five ask about something no policy covers. An answer "
        "counts as unsupported if it cites a policy that doesn't exist, answers a question no policy "
        "covers, or contains figures that aren't in the cited policy."
    )
    ok = results[results["error"] == ""].copy()
    ok["seconds"] = ok["seconds"].astype(float)
    times = ok.groupby("approach")["seconds"].agg(["mean", "median"])
    label_to_key = {v: k for k, v in LABELS.items()}
    table = summary.copy()
    keys = table["approach"].map(label_to_key)
    table["avg_seconds"] = [fmt_time(times.loc[k, "mean"]) for k in keys]
    table["median_seconds"] = [fmt_time(times.loc[k, "median"]) for k in keys]
    table = table.drop(columns=[c for c in ["answered", "failed_calls"] if c in table])
    table = table.rename(columns={
        "approach": "Approach", "model": "Model", "accuracy": "Accuracy",
        "correct_policy_in_scope": "Right policy (covered questions)",
        "correctly_declined_out_of_scope": "Declined uncovered questions",
        "unsupported_rate": "Unsupported answers", "avg_seconds": "Avg. response time",
        "median_seconds": "Median response time", "avg_tokens": "Avg. tokens per question",
    })
    st.dataframe(table, hide_index=True, width="stretch")

    def cell(r) -> str:
        mark = "✓" if str(r["correct"]) == "True" else "✗"
        warn = " ⚠" if str(r["unsupported"]) == "True" else ""
        return f"{mark} {r['policy_title'] or '—'}{warn}"

    ok["cell"] = ok.apply(cell, axis=1)
    per_q = ok.pivot_table(index=["question", "expected_policy"], columns="approach",
                           values="cell", aggfunc="first").reset_index()
    order = tests["question"].tolist() if tests is not None else per_q["question"].tolist()
    per_q["question"] = pd.Categorical(per_q["question"], order, ordered=True)
    per_q = per_q.sort_values("question")
    per_q = per_q.rename(columns={"question": "Question", "expected_policy": "Expected policy",
                                  **{k: LABELS[k] for k in DEFAULT_APPROACHES}})
    per_q["Expected policy"] = per_q["Expected policy"].replace("", "none (not covered)")
    st.caption("Per question: ✓ right policy or correctly declined, ✗ wrong, ⚠ flagged as unsupported.")
    st.dataframe(per_q[["Question", "Expected policy"] + [LABELS[k] for k in DEFAULT_APPROACHES]],
                 hide_index=True, width="stretch")

# --- Conclusion ------------------------------------------------------------------------------
st.header("Comparison and preferred approach")
st.markdown(COMPARISON.read_text(encoding="utf-8") if COMPARISON.exists() else "_comparison.md is missing._")

repo = os.getenv("REPO_URL")
if repo:
    st.caption(f"Source code: {repo}")
