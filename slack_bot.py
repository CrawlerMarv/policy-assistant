"""Slack bot that answers policy questions and names the relevant policy.

Runs locally via Socket Mode (no public URL needed):

    python slack_bot.py

Needs in .env:  SLACK_BOT_TOKEN=xoxb-...   SLACK_APP_TOKEN=xapp-...
Mention the bot in a channel it was invited to:  @Policy Bot How many vacation days do I get?
"""
from __future__ import annotations

import os
import re

from policy_assistant import LABELS, Result, ask, llm, warm_up  # also loads .env

APPROACH = os.getenv("SLACK_APPROACH", "llm_rag")  # the preferred approach from the comparison
EXAMPLE = "How many vacation days do I get per year?"


def clean_question(text: str) -> str:
    """Removes the bot mention (<@U123ABC>) from the message."""
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


def format_reply(r: Result) -> str:
    if r.policy_title:
        body = f"{r.answer}\n\n*Relevant policy:* {r.policy_title}\n> {r.policy_text}"
    else:
        body = f"{r.answer}\n\n_No policy in the database covers this question._"
    footer = f"_{LABELS[r.approach]}, {r.seconds:.2f} s, {r.total_tokens} tokens_"
    return f"{body}\n{footer}"


def answer_text(question: str) -> str:
    if not question:
        return f"Ask me a question about company policy, for example: _{EXAMPLE}_"
    try:
        return format_reply(ask(APPROACH, question))
    except llm.QuotaExceeded:
        return "Today's API quota is used up. Please try again after midnight Pacific time."
    except Exception as e:  # keep the bot alive and tell the user what happened
        return f"Sorry, I couldn't reach the language model ({type(e).__name__}). Please try again."


def handle_mention(event: dict, say) -> None:
    reply = answer_text(clean_question(event.get("text", "")))
    # Answer in the thread if the question was asked in one, otherwise directly in the channel.
    say(text=reply, thread_ts=event.get("thread_ts"))


def main() -> None:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    missing = [k for k in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN") if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")

    print("Building the search index…")
    warm_up([APPROACH])
    llm.warm_up_connection()

    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    app.event("app_mention")(handle_mention)
    print(f"Policy bot is running ({LABELS[APPROACH]}, {llm.PROVIDER}/{llm.MODEL}). Stop with Ctrl+C.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
