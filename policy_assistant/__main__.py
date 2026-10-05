"""Quick manual test:  python -m policy_assistant "Can I work from home on Fridays?" [-a rules llm_rag]"""
import argparse

from . import DEFAULT_APPROACHES, LABELS, APPROACHES, ask, warm_up

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("-a", "--approaches", nargs="+", default=DEFAULT_APPROACHES, choices=list(APPROACHES))
args = parser.parse_args()

warm_up(args.approaches)
for name in args.approaches:
    r = ask(name, args.question)
    print(f"\n=== {LABELS[name]} ===")
    print(f"Answer : {r.answer}")
    print(f"Policy : {r.policy_title or '—'}")
    print(f"Time   : {r.seconds:.2f}s   Tokens: {r.total_tokens} ({r.input_tokens} in / {r.output_tokens} out)")
    if r.retrieved:
        print(f"Hits   : {', '.join(r.retrieved)}")
