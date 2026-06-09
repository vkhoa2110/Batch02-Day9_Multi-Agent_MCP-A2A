"""CLI entrypoint for the Day09 Supervisor-Workers assignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Lab_Assignment.supervisor import SupervisorAgent, format_report


DEFAULT_QUESTION = "Tàng trữ ma túy bị xử phạt như thế nào?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day08 RAG Supervisor-Workers demo.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUESTION, help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final sources to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supervisor = SupervisorAgent()
    result = supervisor.run(args.query, top_k=args.top_k)
    print(format_report(result))


if __name__ == "__main__":
    main()
