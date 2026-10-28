"""
The LangGraph workflow:

    START -> planner -> +-> data_analyst -+-> writer -> END
                        +-> knowledge ----+

data_analyst and knowledge run IN PARALLEL (both only need the plan), and the
writer waits for both. Each node writes different state keys, so there are no
conflicts to merge.
"""
from __future__ import annotations

import sqlite3
import time
from functools import lru_cache
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from rca import analytics as A
from rca.agents import RCAState, data_analyst, knowledge, planner, writer
from rca.knowledge import KnowledgeBase
from rca.llm import BaseLLM, get_llm


def build_graph(llm: BaseLLM, con: sqlite3.Connection, kb: KnowledgeBase):
    g = StateGraph(RCAState)
    g.add_node("planner", lambda s: planner(s, llm, con))
    g.add_node("data_analyst", lambda s: data_analyst(s, llm, con))
    g.add_node("knowledge", lambda s: knowledge(s, kb))
    g.add_node("writer", lambda s: writer(s, llm))

    g.add_edge(START, "planner")
    g.add_edge("planner", "data_analyst")           # fan out ...
    g.add_edge("planner", "knowledge")
    g.add_edge(["data_analyst", "knowledge"], "writer")   # ... and join: writer waits for both
    g.add_edge("writer", END)
    return g.compile()


@lru_cache(maxsize=1)
def default_kb() -> KnowledgeBase:
    """Build the knowledge base once per process (embedding the documents takes a moment)."""
    return KnowledgeBase()


def investigate(question: str, llm: BaseLLM | None = None, db_path: str | Path = A.DEFAULT_DB,
                kb: KnowledgeBase | None = None) -> dict:
    llm = llm or get_llm()
    kb = kb or default_kb()
    # check_same_thread=False: LangGraph runs parallel nodes in worker threads; all access is read-only.
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
    try:
        t0 = time.perf_counter()
        state = build_graph(llm, con, kb).invoke({"question": question})
        state["usage"] = {**llm.usage_summary(), "total_seconds": round(time.perf_counter() - t0, 2),
                          "model": llm.model, "retrieval": kb.mode}
        return state
    finally:
        con.close()
