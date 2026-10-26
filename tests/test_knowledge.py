"""
Retrieval tests. They run in BM25-only mode, so they need no API key and always
give the same answer. Each query is phrased the way an engineer (or the agent)
would ask, and must find the documents a human expert would pick.
"""
import pytest

from rca.knowledge import KnowledgeBase, load_chunks, reciprocal_rank_fusion, tokenize


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(use_embeddings=False)


def top_docs(kb, query, k=3, **kw):
    return [r["doc_id"] for r in kb.search(query, k=k, **kw)]


def test_tokenizer_keeps_equipment_codes():
    assert {"p-301", "bleach_temp", "h2o2"} <= set(tokenize("Check P-301, bleach_temp and H2O2 dose"))


def test_sops_are_chunked_by_section_and_incidents_kept_whole():
    chunks = load_chunks()
    sop_sections = [c.section for c in chunks if c.doc_id == "SOP-BL-01"]
    assert "Troubleshooting low brightness" in sop_sections
    assert sum(c.doc_id == "INC-2025-014" for c in chunks) == 1
    assert all(c.text.startswith(c.title) for c in chunks)          # contextual header


@pytest.mark.parametrize("query, expected", [
    ("peroxide pump P-301 unsteady flow, brightness dropped", {"INC-2025-014", "SOP-BL-01"}),
    ("bleach tower temperature low steam valve TV-210", {"SOP-BL-02", "INC-2025-031"}),
    ("pH alarm high but brightness and quality normal", {"SOP-INS-01", "INC-2026-002"}),
    ("WIS defect rate rising slowly, holes and spots", {"SOP-PM-03", "INC-2025-047"}),
    ("brightness low but no process variable deviated, new pulp supplier", {"SOP-QA-02", "INC-2026-011"}),
    ("effluent TSS high, disc filter differential pressure rising", {"SOP-ENV-01", "INC-2025-022"}),
])
def test_finds_the_right_documents(kb, query, expected):
    assert set(top_docs(kb, query, k=2)) & expected, top_docs(kb, query, k=5)


def test_doc_type_filter(kb):
    assert all(r["doc_type"] == "incident" for r in kb.search("brightness", k=5, doc_type="incident"))


def test_rrf_rewards_agreement():
    # chunk 1 is 2nd in BOTH lists; chunks 0 and 2 are each 1st in one list but missing from the other
    scores = reciprocal_rank_fusion([[0, 1], [2, 1]])
    assert max(scores, key=scores.get) == 1
