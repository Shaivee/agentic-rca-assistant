"""
Command-line entry point.

    python -m rca.cli "Why did batch B-1034 fail the brightness spec?"
    python -m rca.cli "Batch B-1087 had high pH alarms. Is quality affected?" --json
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap

from rca.graph import investigate


def print_report(state: dict) -> None:
    r, u = state["report"], state["usage"]
    w = lambda s, ind="  ": textwrap.fill(str(s), 100, initial_indent=ind, subsequent_indent=ind)  # noqa: E731
    print("\n" + "=" * 100)
    print(f"ROOT-CAUSE REPORT  |  confidence: {r['confidence'].upper()}  |  category: {r['root_cause_category']}"
          f"  |  equipment: {r['equipment']}  |  quality impact: {r['quality_impact']}")
    print("=" * 100)
    print("Problem:\n" + w(r["problem_statement"]))
    print("Affected batches: " + ", ".join(r["affected_batches"]))
    print("Root cause:\n" + w(r["root_cause"]))
    print("Evidence:")
    for e in r["evidence"]:
        print(w(f"[{e['kind']}] {e['claim']}  (source: {e['source']})", "  - "))
    if r["ruled_out"]:
        print("Ruled out:")
        for x in r["ruled_out"]:
            print(w(x, "  - "))
    print("Corrective actions:")
    for x in r["corrective_actions"]:
        print(w(x, "  - "))
    print("Preventive actions:")
    for x in r["preventive_actions"]:
        print(w(x, "  - "))
    print("References: " + ", ".join(r["references"]))
    print("-" * 100)
    print("Tools called: " + " -> ".join(t["tool"] for t in state["tool_trace"]))
    print(f"Documents retrieved: {', '.join(dict.fromkeys(d['doc_id'] for d in state['documents']))}  (retrieval: {u['retrieval']})")
    print(f"Model: {u['model']} | LLM calls: {u['llm_calls']} | tokens in/out: {u['prompt_tokens']}/"
          f"{u['completion_tokens']} | time: {u['total_seconds']} s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic root-cause analysis for the simulated paper mill")
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="print the full state as JSON")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="[warning] %(message)s")
    state = investigate(args.question)
    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        print_report(state)


if __name__ == "__main__":
    main()
