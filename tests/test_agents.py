"""
Agent tests with a ScriptedLLM: the LLM's replies are pre-written, but everything
else is real - the LangGraph wiring, the tool calls against the plant database,
retrieval, and Pydantic validation. This checks the plumbing without an API key.
"""
import json
import sqlite3

import pytest

from rca import tools
from rca.graph import investigate
from rca.knowledge import KnowledgeBase
from rca.llm import ChatResult, ScriptedLLM, ToolCall
from rca.schemas import InvestigationPlan
from rca.simulation import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="run `python -m rca.simulation` first")

PLAN = {"batch_id": "B-1034", "symptom": "low brightness",
        "hypotheses": ["peroxide dosing pump problem", "low bleach temperature", "pulp lot change"]}
FINDINGS = {"summary": "Peroxide dose fell to ~15 kg/t and led brightness by 30 min; P-301 diaphragm torn.",
            "evidence": [{"kind": "data", "claim": "peroxide_dose median 15.3 vs 24.9 kg/t", "source": "detect_anomalies"},
                         {"kind": "timing", "claim": "peroxide leads brightness by 30 min (r=0.99)", "source": "find_drivers"}],
            "candidate_causes": ["P-301 dosing pump failure"], "ruled_out": ["Conveyor C-12 maintenance"]}
REPORT = {"problem_statement": "B-1034 brightness 82.9 %ISO, below 84 spec.", "affected_batches": ["B-1034"],
          "quality_impact": True, "root_cause": "P-301 diaphragm tore, cutting peroxide dose.",
          "root_cause_category": "dosing_equipment", "equipment": "P-301", "confidence": "high",
          "evidence": FINDINGS["evidence"], "ruled_out": ["Conveyor C-12"],
          "corrective_actions": ["Replace diaphragm"], "preventive_actions": ["Add P-301 to inspection route"],
          "references": ["INC-2025-014", "SOP-BL-01"]}


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(use_embeddings=False)


def scripted_run():
    return ScriptedLLM([
        json.dumps(PLAN),                                                   # planner
        ChatResult(tool_calls=[                                             # analyst, round 1: two tool calls
            ToolCall("c1", "detect_anomalies", {"batch_id": "B-1034"}),
            ToolCall("c2", "find_drivers", {"effect_tag": "brightness_online",
                                            "start": "2026-09-06 04:00", "end": "2026-09-06 22:00"})]),
        "Peroxide dose dropped and leads brightness by 30 minutes.",       # analyst, round 2: done
        json.dumps(FINDINGS),                                               # analyst -> structured findings
        json.dumps(REPORT),                                                 # writer
    ])


def test_full_graph_runs_end_to_end(kb):
    llm = scripted_run()
    state = investigate("Why did batch B-1034 fail?", llm=llm, kb=kb)

    assert state["batch"]["result"] == "FAIL"                 # planner fetched facts from the DB
    tools_called = [t["tool"] for t in state["tool_trace"]]
    assert tools_called == ["detect_anomalies", "find_drivers"]
    flagged = [r["tag"] for r in state["tool_trace"][0]["result"]]
    assert "peroxide_dose" in flagged                          # real tool output, not scripted
    assert state["tool_trace"][1]["result"][0]["cause_tag"] == "peroxide_dose"
    assert {"INC-2025-014", "SOP-BL-01"} & {d["doc_id"] for d in state["documents"]}
    assert state["report"]["root_cause_category"] == "dosing_equipment"
    assert state["usage"]["llm_calls"] == 5


def test_tool_results_are_sent_back_to_the_llm(kb):
    llm = scripted_run()
    investigate("Why did batch B-1034 fail?", llm=llm, kb=kb)
    second_analyst_call = llm.seen[2]                          # the call after the tools ran
    tool_msgs = [m for m in second_analyst_call if m["role"] == "tool"]
    assert len(tool_msgs) == 2 and "peroxide_dose" in tool_msgs[0]["content"]


def test_structured_output_retries_once_on_invalid_json():
    llm = ScriptedLLM(['{"batch_id": "B-1"}',                  # missing required fields -> invalid
                       json.dumps(PLAN)])                       # corrected
    plan = llm.structured([{"role": "user", "content": "x"}], InvestigationPlan)
    assert plan.batch_id == "B-1034"
    assert "invalid" in llm.seen[1][-1]["content"]             # the model was shown its error


def test_tool_errors_are_returned_not_raised():
    con = sqlite3.connect(DB_PATH)
    out = tools.execute(con, "batch_context", {"batch_id": "B-9999"})
    assert "error" in out and "Unknown batch_id" in out["error"]


@pytest.mark.parametrize("bad", ["DELETE FROM events", "SELECT 1; DROP TABLE tags",
                                 "UPDATE tags SET unit='x'", "PRAGMA table_info(tags)"])
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(ValueError):
        tools.check_sql(bad)


def test_sql_guard_allows_reads_and_limits_rows():
    con = sqlite3.connect(DB_PATH)
    out = tools.run_sql(con, "SELECT * FROM process_readings")
    assert len(out["rows"]) == tools.MAX_SQL_ROWS and out["truncated"]


# ---------------------------------------------------------------------------
# Robustness: what happens when the model produces output that cannot be parsed
# ---------------------------------------------------------------------------
from rca.llm import LLMOutputError  # noqa: E402


def test_analyst_recovers_after_an_unparseable_reply(kb):
    llm = ScriptedLLM([
        json.dumps(PLAN),
        LLMOutputError("garbled tool call"),                                 # round 1 fails
        ChatResult(tool_calls=[ToolCall("c1", "detect_anomalies", {"batch_id": "B-1034"})]),
        "Peroxide dose dropped.",
        json.dumps(FINDINGS), json.dumps(REPORT),
    ])
    state = investigate("Why did batch B-1034 fail?", llm=llm, kb=kb)
    assert [t["tool"] for t in state["tool_trace"]] == ["detect_anomalies"]
    assert any("could not be parsed" in m["content"] for m in llm.seen[1] if m["role"] == "user")


def test_analyst_stops_early_but_still_reports(kb):
    llm = ScriptedLLM([json.dumps(PLAN), LLMOutputError("x"), LLMOutputError("y"),
                       json.dumps(FINDINGS), json.dumps(REPORT)])
    state = investigate("Why did batch B-1034 fail?", llm=llm, kb=kb)
    assert state["tool_trace"] == [] and state["report"]["root_cause_category"]


def test_retry_uses_a_different_temperature(monkeypatch):
    """A 400 'failed_generation' must be retried with randomness, not repeated identically."""
    import httpx2
    from rca.llm import OpenAICompatibleLLM, RETRY_TEMPERATURE
    monkeypatch.setenv("GROQ_API_KEY", "test")
    temps = []

    def handler(request):
        temps.append(json.loads(request.content)["temperature"])
        if len(temps) == 1:
            return httpx2.Response(400, json={"error": {"message": "Parsing failed.", "type": "invalid_request_error",
                                                        "code": "tool_use_failed", "failed_generation": "<|garbled|>"}})
        return httpx2.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                          "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "ok"}}]})

    llm = OpenAICompatibleLLM(provider="groq")
    llm.client = llm.client.with_options(http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
                                         max_retries=0)
    assert llm.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert temps == [0.0, RETRY_TEMPERATURE]
