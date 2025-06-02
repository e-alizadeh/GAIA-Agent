"""
Run ONE GAIA task locally through the full GAIAAgent pipeline.
Shows debug prints and, if an attachment exists, saves it to disk.

Usage:
    python debug_one_task.py <task_id>
"""

import sys
import textwrap

import requests

from app import DEFAULT_API_URL, GAIAAgent


def fetch_question_row(task_id: str, api: str = DEFAULT_API_URL) -> dict[str, str]:
    """Return the question dict for the given task_id (raises if not found)."""
    resp = requests.get(f"{api}/questions", timeout=15)
    resp.raise_for_status()
    for row in resp.json():
        if row["task_id"] == task_id:
            print(f"\n\n{row}\n\n")
            return row
    raise ValueError(f"Task ID '{task_id}' not present in /questions endpoint.")


def main(task_id: str) -> None:
    agent = GAIAAgent()

    row = fetch_question_row(task_id)

    print("=" * 90)
    print(f"QUESTION  ({task_id})")
    print(textwrap.fill(row["question"], width=90))
    print("=" * 90)

    answer = agent(row["question"], task_id=task_id)
    print("\nFINAL ANSWER  →", answer)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage:  python debug_one_task.py <task_id>")
    main(sys.argv[1].strip())
