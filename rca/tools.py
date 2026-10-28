"""
Tool layer: exposes the Phase 2 analytics (plus a guarded SQL tool) to the LLM.

Tool calling, in short:
  1. We send the LLM a JSON-schema description of each function (TOOL_SCHEMAS).
  2. The LLM replies "please call detect_anomalies with batch_id='B-1034'".
  3. OUR code runs the function (`execute`) and sends the result back.
  The LLM never executes anything itself; it only asks.

Results are trimmed before going back to the LLM: shorter context = lower cost,
fewer rate-limit problems on free tiers, and less noise for the model to wade through.
"""
from __future__ import annotations

import json
import re
import sqlite3

import pandas as pd

from rca import analytics as A

MAX_SQL_ROWS = 50


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_TS = {"type": "string", "description": "Timestamp 'YYYY-MM-DD HH:MM'"}
_BATCH = {"type": "string", "description": "Batch id, e.g. 'B-1034'"}

TOOL_SCHEMAS = [
    _fn("batch_context", "Batch time window, quality result, failed specs, pulp lot and whether the supplier is new.",
        {"batch_id": _BATCH}, ["batch_id"]),
    _fn("detect_anomalies", "Robust z-score of every sensor during the batch vs its previous 7 days. "
        "Returns flagged tags and the strongest deviations.", {"batch_id": _BATCH}, ["batch_id"]),
    _fn("compare_batches", "Compare the batch with the last 12 PASS batches: effect size per process tag "
        "and per quality result.", {"batch_id": _BATCH}, ["batch_id"]),
    _fn("find_drivers", "Rank process tags by lagged cross-correlation with an effect tag "
        "(e.g. brightness_online, defect_rate, effluent_tss) in a time window. Use a window a few hours "
        "wider than the batch so the lag is visible.",
        {"effect_tag": {"type": "string"}, "start": _TS, "end": _TS}, ["effect_tag", "start", "end"]),
    _fn("cross_correlate", "Lag (minutes) and correlation between one cause tag and one effect tag in a window.",
        {"cause_tag": {"type": "string"}, "effect_tag": {"type": "string"}, "start": _TS, "end": _TS},
        ["cause_tag", "effect_tag", "start", "end"]),
    _fn("correlate_events", "Maintenance logs, operator notes, alarms and material changes from 24 h before "
        "`start` to 12 h after `end`, each with hours_from_start and the sensor tags its equipment links to.",
        {"start": _TS, "end": _TS,
         "event_types": {"type": "array", "items": {"type": "string",
                         "enum": ["maintenance", "operator_note", "alarm", "material_change"]}}},
        ["start", "end"]),
    _fn("run_sql", "Read-only SQL (SQLite) for anything the other tools do not cover. Tables: tags, "
        "process_readings(ts, tag, value), batches, pulp_lots, quality_results, events. "
        f"Single SELECT only; at most {MAX_SQL_ROWS} rows are returned.",
        {"query": {"type": "string"}}, ["query"]),
]


# ---------------------------------------------------------------------------
# SQL guardrail (defence in depth: the connection is ALSO read-only)
# ---------------------------------------------------------------------------
FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b", re.I)


def check_sql(query: str) -> str:
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise ValueError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", q, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    if FORBIDDEN_SQL.search(q):
        raise ValueError("Query contains a forbidden keyword.")
    return q


def run_sql(con: sqlite3.Connection, query: str) -> dict:
    q = check_sql(query)
    df = pd.read_sql(f"SELECT * FROM ({q}) LIMIT {MAX_SQL_ROWS + 1}", con)
    return {"rows": df.head(MAX_SQL_ROWS).to_dict(orient="records"), "truncated": len(df) > MAX_SQL_ROWS}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def execute(con: sqlite3.Connection, name: str, args: dict) -> dict | list:
    """Run one tool call. Errors are RETURNED to the LLM (not raised) so it can
    correct itself, e.g. fix a bad timestamp or a misspelled tag."""
    try:
        if name == "batch_context":
            return A.batch_context(con, args["batch_id"])
        if name == "detect_anomalies":
            rows = A.detect_anomalies(con, args["batch_id"])
            keep = [r for r in rows if r["flagged"]] or rows[:3]
            return [{k: r[k] for k in ("tag", "unit", "baseline_median", "batch_median", "median_robust_z",
                                        "pct_readings_anomalous", "first_anomalous_ts", "readings_outside_limits")}
                    for r in keep]
        if name == "compare_batches":
            cmp = A.compare_batches(con, args["batch_id"])
            return {"process_top5": cmp["process"][:5], "quality": cmp["quality"],
                    "n_reference_batches": len(cmp["reference_batches"])}
        if name == "find_drivers":
            return A.find_drivers(con, args["effect_tag"], args["start"], args["end"])
        if name == "cross_correlate":
            r = A.cross_correlate(con, args["cause_tag"], args["effect_tag"], args["start"], args["end"])
            r.pop("curve_every_15_min")
            return r
        if name == "correlate_events":
            events = A.correlate_events(con, args["start"], args["end"], event_types=args.get("event_types"))
            return [{k: e[k] for k in ("event_id", "ts", "event_type", "equipment", "description",
                                       "hours_from_start", "linked_tags")} for e in events][:25]
        if name == "run_sql":
            return run_sql(con, args["query"])
        return {"error": f"Unknown tool {name}"}
    except Exception as err:                       # noqa: BLE001 - report any failure back to the model
        return {"error": f"{type(err).__name__}: {err}"}


def to_json(result) -> str:
    return json.dumps(result, default=str)
