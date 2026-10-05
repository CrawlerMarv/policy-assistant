"""Shared LLM call used by approaches 2 and 3.

Provider is chosen in .env:  LLM_PROVIDER=gemini  (free tier via Google AI Studio)
                             LLM_PROVIDER=anthropic
"""
from __future__ import annotations

import json
import os
import re
import time

from .data import Policy, Result, match_title

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5-20251001", "gemini": "gemini-3.5-flash-lite"}
if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER must be one of {list(DEFAULT_MODELS)}, got {PROVIDER!r}")
MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS[PROVIDER]
MAX_TOKENS = 300

JSON_FORMAT = (
    'Respond with a single JSON object and nothing else (no markdown):\n'
    '{"answer": "<1-2 sentences>", "policy_title": "<exact policy title or null>", "covered": <true|false>}'
)

GROUNDED_SYSTEM = """You are a company policy assistant. Answer the employee's question using ONLY the policies listed below.

Rules:
- If a listed policy answers the question, answer in 1-2 sentences and give its exact title.
- If a policy is relevant but does not state the specific detail asked for, cite it and say that the detail is not specified.
- Never add facts, numbers, or conditions that are not in the policy text.
- If no listed policy covers the question, set "covered" to false and "policy_title" to null, and say that the policies don't cover it.

{json_format}

Policies:
{policies}"""

# Optional baseline: the LLM gets no policy data at all ("closed book").
CLOSED_BOOK_SYSTEM = """You are a company policy assistant. Answer the employee's question about company policy
in 1-2 sentences and name the policy that applies.

{json_format}"""

_client = None


def client():
    global _client
    if _client is None:
        if PROVIDER == "gemini":
            from google import genai  # reads GEMINI_API_KEY from the environment
            _client = genai.Client()
        else:
            from anthropic import Anthropic  # reads ANTHROPIC_API_KEY from the environment
            _client = Anthropic()
    return _client


RETRY_CODES = {429, 500, 503}  # rate limit, server error, "model overloaded"
MAX_ATTEMPTS = 6


class QuotaExceeded(RuntimeError):
    """Daily (or otherwise persistent) quota is used up – retrying today is pointless."""


def warm_up_connection() -> None:
    """Opens the HTTPS connection with a metadata call that uses no generation quota."""
    try:
        if PROVIDER == "gemini":
            client().models.get(model=MODEL)
        else:
            client().models.retrieve(MODEL)
    except Exception as e:  # warm-up is optional
        print(f"(warm-up skipped: {str(e)[:100]})")


def _call_anthropic(system: str, question: str) -> tuple[str, int, int, float]:
    resp = client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens, 0.0  # SDK retries internally


def _call_gemini(system: str, question: str) -> tuple[str, int, int, float]:
    from google.genai import errors, types

    cfg = dict(system_instruction=system, max_output_tokens=MAX_TOKENS,
               response_mime_type="application/json")
    if MODEL.startswith("gemini-2"):
        # 2.x models: thinking switched off, deterministic sampling.
        cfg["temperature"] = 0
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    else:
        # 3.x models: no temperature setting; thinking can't be switched off, "low" is the
        # lowest level. Thinking tokens share the output budget, so it is raised and the
        # thought tokens are counted as output tokens below.
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        cfg["max_output_tokens"] = 4096
    waited = 0.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client().models.generate_content(
                model=MODEL, contents=question, config=types.GenerateContentConfig(**cfg))
            break
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                raise QuotaExceeded(f"daily free-tier quota for {MODEL} used up") from e
            if e.code == 400 and "thinking" in str(e).lower() and "thinking_config" in cfg:
                cfg.pop("thinking_config")  # model doesn't support this thinking setting
                continue
            if e.code == 429 and attempt == MAX_ATTEMPTS - 1:
                raise QuotaExceeded(f"quota for {MODEL} still exhausted after retries") from e
            if e.code in RETRY_CODES and attempt < MAX_ATTEMPTS - 1:
                pause = 10 * (attempt + 1)
                print(f"  ({e.code} from Gemini, retrying in {pause}s)")
                time.sleep(pause)
                waited += pause
                continue
            raise
    u = resp.usage_metadata
    output = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    return resp.text or "", u.prompt_token_count or 0, output, waited


def grounded_system(policies: list[Policy] | tuple[Policy, ...]) -> str:
    return GROUNDED_SYSTEM.format(json_format=JSON_FORMAT,
                                  policies="\n".join(p.as_context() for p in policies))


def closed_book_system() -> str:
    return CLOSED_BOOK_SYSTEM.format(json_format=JSON_FORMAT)


def parse_response(text: str) -> tuple[str, str | None]:
    """Returns (answer, policy_title). policy_title is None if the model says nothing covers it."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        data = json.loads(m.group(0) if m else cleaned)
    except (json.JSONDecodeError, AttributeError):
        # Unparseable output: keep the raw text, no policy. Evaluation flags this as unsupported.
        return cleaned, "<unparsed>"
    title = data.get("policy_title")
    if isinstance(title, str) and title.strip().lower() in {"", "null", "none"}:
        title = None
    if data.get("covered") is False:
        title = None
    return str(data.get("answer", "")).strip(), match_title(title)


def ask_llm(approach: str, system: str, question: str, retrieved: list[str] | None = None) -> Result:
    call = _call_gemini if PROVIDER == "gemini" else _call_anthropic
    text, input_tokens, output_tokens, waited = call(system, question)
    answer, title = parse_response(text)
    return Result(
        approach=approach,
        question=question,
        answer=answer,
        policy_title=title,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retrieved=retrieved or [],
        retry_wait=waited,
    )
