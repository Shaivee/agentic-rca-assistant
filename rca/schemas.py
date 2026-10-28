"""
Pydantic schemas for everything the LLM produces.

Forcing the LLM into a fixed structure means:
  * downstream code never has to parse free text,
  * every report has the same fields (root cause, confidence, evidence ...),
  * evaluation can compare `root_cause_category` with the ground truth automatically.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RootCauseCategory = Literal[
    "dosing_equipment", "steam_supply", "sensor_fault", "equipment_wear",
    "raw_material", "filtration", "other", "undetermined",
]


class InvestigationPlan(BaseModel):
    """What the Planner extracts from the user's question."""
    batch_id: str | None = Field(None, description="Batch id like 'B-1034' if the question names one")
    symptom: str = Field(description="The problem in a few words, e.g. 'low brightness', 'high pH alarms'")
    hypotheses: list[str] = Field(description="3-5 plausible causes worth checking, short phrases")


class Evidence(BaseModel):
    kind: Literal["data", "timing", "record", "document"] = Field(
        description="data = abnormal measurement; timing = lag/sequence; record = log/event; document = SOP/incident")
    claim: str = Field(description="One factual sentence, with numbers where available")
    source: str = Field(description="Tool name, event id, tag name, or document id the claim comes from")


class DataFindings(BaseModel):
    """What the Data Analyst concludes from its tool calls."""
    summary: str
    evidence: list[Evidence]
    candidate_causes: list[str] = Field(description="Most likely first")
    ruled_out: list[str] = Field(default_factory=list, description="Things checked and found NOT to be the cause")


class RCAReport(BaseModel):
    """The final root-cause report."""
    problem_statement: str
    affected_batches: list[str]
    quality_impact: bool = Field(description="False if product quality was actually fine (e.g. a sensor fault)")
    root_cause: str = Field(description="One or two sentences naming the cause and the mechanism")
    root_cause_category: RootCauseCategory
    equipment: str | None = Field(None, description="Equipment tag responsible, e.g. 'P-301', or null")
    confidence: Literal["high", "medium", "low"]
    evidence: list[Evidence]
    ruled_out: list[str] = Field(default_factory=list)
    corrective_actions: list[str]
    preventive_actions: list[str]
    references: list[str] = Field(default_factory=list, description="Ids of SOPs / past incidents used")
