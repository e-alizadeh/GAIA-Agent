import argparse
import textwrap
from typing import Any

import requests

from app import DEFAULT_API_URL, GAIAAgent


def fetch_question_row(task_id: str, api: str = DEFAULT_API_URL) -> dict[str, Any]:
    """Return the question dict associated with *task_id* (raises if not found)."""
    resp = requests.get(f"{api}/questions", timeout=15)
    resp.raise_for_status()
    for row in resp.json():
        if row["task_id"] == task_id:
            return row
    raise ValueError(f"task_id '{task_id}' not present in /questions.")


def run_one(task_id: str | None, question: str | None) -> None:
    agent = GAIAAgent()

    if task_id:
        row = fetch_question_row(task_id)
        question = row["question"]
        print(f"\n{row}\n")  # show full row incl. metadata

    # --- show pretty question
    print("=" * 90)
    print(f"QUESTION ({task_id or 'adhoc'})")
    print(textwrap.fill(question or "", width=90))
    print("=" * 90)

    assert question is not None, "Internal error: question was None"
    answer = agent(question, task_id=task_id)
    print(f"\nFINAL ANSWER --> {answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GAIAAgent query locally.")
    parser.add_argument("--task_id", help="GAIA task_id to fetch & run")
    parser.add_argument("question", nargs="?", help="Ad-hoc question text (positional)")

    ns = parser.parse_args()

    # mutual-exclusion checks
    if ns.task_id and ns.question:
        parser.error("Provide either --task_id OR a question, not both.")
    if ns.task_id is None and ns.question is None:
        parser.error("You must supply a GAIA --task_id or a question.")

    return ns


if __name__ == "__main__":
    args = parse_args()
    run_one(task_id=args.task_id, question=args.question)
