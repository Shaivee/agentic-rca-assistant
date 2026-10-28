"""
The agents (graph nodes). Each node reads the shared state and returns ONLY the
keys it updates; LangGraph merges them into the state.

Design rule used throughout: the LLM does language and judgement; plain code
does facts. E.g. the Planner LLM extracts the batch id from the question, but the
batch's real time window comes from the database, never from the model.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, TypedDict

from rca import analytics as A
from rca.knowledge import KnowledgeBase
from rca.llm import BaseLLM, LLMOutputError
from rca.schemas import DataFindings, InvestigationPlan, RCAReport
from rca.tools import TOOL_SCHEMAS, execute, to_json

log = logging.getLogger("rca")

MAX_TOOL_ROUNDS = 8
MAX_OUTPUT_FAILURES = 2   # consecutive unparseable replies before the analyst stops

# Which online signal to analyse for each failed quality spec.
SPEC_TO_TAG = {"brightness_iso": "brightness_online", "defects_per_km": "defect_rate",
               "effluent_tss": "effluent_tss"}


class RCAState(TypedDict, total=False):
    question: str
    plan: dict
    batch: dict
    tool_trace: list[dict]
    findings: dict
    documents: list[dict]
    report: dict


# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You plan root-cause investigations for a paper mill (bleach plant, paper machine, effluent).
From the engineer's question, extract the batch id (if any) and the symptom in a few words, and list
3-5 hypotheses that span DIFFERENT kinds of cause, for example: chemical dosing equipment, temperature
or steam supply, instrument/sensor fault, raw material change, equipment wear, filtration."""


def planner(state: RCAState, llm: BaseLLM, con: sqlite3.Connection) -> dict:
    plan = llm.structured([{"role": "system", "content": PLANNER_PROMPT},
                           {"role": "user", "content": state["question"]}], InvestigationPlan, node="planner")
    update: dict[str, Any] = {"plan": plan.model_dump()}
    if plan.batch_id:
        update["batch"] = A.batch_context(con, plan.batch_id)     # facts from the DB, not the LLM
    return update


# ---------------------------------------------------------------------------
# 2. Data Analyst (tool-calling agent)
# ---------------------------------------------------------------------------
ANALYST_PROMPT = f"""You are a senior process engineer investigating a quality problem with data tools.

Suggested method:
1. detect_anomalies and compare_batches for the batch: which variables deviated, and how strongly?
2. find_drivers for the affected quality signal ({", ".join(f"{k} -> {v}" for k, v in SPEC_TO_TAG.items())})
   over a window from ~4 h before the batch start to ~2 h after its end: which variable LEADS the effect,
   and by how many minutes?
3. correlate_events around the batch: maintenance logs, operator notes, alarms, material changes.
4. Use run_sql only for something the other tools cannot answer.

Evidence rules (SOP-RCA-00):
- A cause needs at least two independent kinds of evidence: data (an anomaly), timing (it precedes the
  effect with a plausible delay), records (a log, note or material change that explains it).
- Correlation alone is not proof. Ignore events with no physical link to the deviating variables.
- An alarm with normal product quality may be a SENSOR fault, not a process fault.
- If no process variable deviated, look at raw materials (pulp lot / supplier).

Call tools until you have enough evidence (at most {MAX_TOOL_ROUNDS} rounds), then reply WITHOUT tool calls
with a concise plain-text summary of what you found, including numbers, lags and event ids."""


def data_analyst(state: RCAState, llm: BaseLLM, con: sqlite3.Connection) -> dict:
    context = {"question": state["question"], "plan": state.get("plan"), "batch": state.get("batch")}
    messages = [{"role": "system", "content": ANALYST_PROMPT},
                {"role": "user", "content": to_json(context)}]
    trace: list[dict] = []

    failures = 0
    for _ in range(MAX_TOOL_ROUNDS):
        try:
            result = llm.chat(messages, tools=TOOL_SCHEMAS, node="data_analyst")
        except LLMOutputError:
            # The model kept producing unreadable tool calls. Tell it, and give it another turn;
            # after repeated failures, stop investigating and summarise the evidence gathered so far.
            failures += 1
            if failures >= MAX_OUTPUT_FAILURES:
                log.warning("Data analyst stopped early after %d unparseable outputs.", failures)
                break
            messages.append({"role": "user", "content":
                             "Your last reply could not be parsed. Call ONE of the provided tools with valid "
                             "JSON arguments, or reply in plain text if you have enough evidence."})
            continue
        failures = 0
        messages.append(result.as_message())
        if not result.tool_calls:
            break
        for call in result.tool_calls:
            output = execute(con, call.name, call.arguments)
            trace.append({"tool": call.name, "arguments": call.arguments, "result": output})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": to_json(output)})

    final_text = messages[-1]["content"] if messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls") \
        else "(no final summary from the analyst; use the tool transcript)"

    # Turn the investigation into structured findings. We pass a compact transcript as plain text
    # (rather than the raw tool-call history) so this works the same on every provider.
    transcript = "\n\n".join(f"TOOL {t['tool']}({to_json(t['arguments'])}) ->\n{to_json(t['result'])}" for t in trace)
    findings = llm.structured([
        {"role": "system", "content": "Convert an investigation into structured findings. "
                                      "Use only facts present in the transcript."},
        {"role": "user", "content": f"Question: {state['question']}\n\nTool transcript:\n{transcript}\n\n"
                                    f"Analyst summary:\n{final_text}"},
    ], DataFindings, node="data_analyst")
    return {"tool_trace": trace, "findings": findings.model_dump()}


# ---------------------------------------------------------------------------
# 3. Knowledge (retrieval; no LLM call needed)
# ---------------------------------------------------------------------------
def knowledge(state: RCAState, kb: KnowledgeBase, max_docs: int = 6) -> dict:
    """Search once for the symptom and once per hypothesis, so the documents cover
    every cause the planner considered, not just the most obvious one."""
    plan = state.get("plan") or {}
    queries = [plan.get("symptom") or state["question"]] + list(plan.get("hypotheses", []))
    best: dict[str, dict] = {}
    for q in queries:
        for hit in kb.search(q, k=3):
            prev = best.get(hit["chunk_id"])
            if prev is None:
                best[hit["chunk_id"]] = {**hit, "hits": 1, "matched_queries": [q]}
            else:
                prev["hits"] += 1
                prev["matched_queries"].append(q)
                prev["score"] = max(prev["score"], hit["score"])
    ranked = sorted(best.values(), key=lambda d: (d["hits"], d["score"]), reverse=True)[:max_docs]
    return {"documents": [{k: d[k] for k in ("doc_id", "title", "section", "text", "matched_queries")}
                          for d in ranked]}


# ---------------------------------------------------------------------------
# 4. RCA Writer
# ---------------------------------------------------------------------------
WRITER_PROMPT = """You write root-cause analysis reports for a paper mill, following SOP-RCA-00.
Rules:
- Base every claim on the data findings or the documents provided. Do not invent numbers or events.
- Past incidents are context, not proof: the current data must support the root cause.
- quality_impact is false when product quality was within spec (e.g. an alarm caused by a faulty sensor).
- confidence: high = data + timing + records agree; medium = two kinds; low = one kind or conflicting.
  If the evidence is insufficient, say so and use root_cause_category "undetermined".
- references: only ids of documents you were given (e.g. SOP-BL-01, INC-2025-014)."""


def writer(state: RCAState, llm: BaseLLM) -> dict:
    docs = "\n\n".join(f"[{d['doc_id']} | {d['section']}]\n{d['text']}" for d in state.get("documents", []))
    user = (f"Question: {state['question']}\n\n"
            f"Batch: {json.dumps(state.get('batch'), default=str)}\n\n"
            f"Data findings: {json.dumps(state.get('findings'))}\n\n"
            f"Documents:\n{docs}")
    report = llm.structured([{"role": "system", "content": WRITER_PROMPT},
                             {"role": "user", "content": user}], RCAReport, node="writer")
    return {"report": report.model_dump()}
