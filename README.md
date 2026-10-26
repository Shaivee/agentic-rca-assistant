# Agentic Root-Cause Analysis Assistant

A multi-agent system that investigates quality incidents in a (simulated) paper mill the way a
senior process engineer would: it queries process data, runs RCA analytics (anomaly detection,
batch comparison, lagged cross-correlation, event correlation), retrieves SOPs and past incidents,
and writes an evidence-backed root-cause report.

> Work in progress — built in phases. See the roadmap below.

## Roadmap
- [x] Phase 1 — Synthetic plant data with 6 injected faults and hidden ground truth
- [x] Phase 2 — RCA analytics toolkit (anomalies, batch comparison, lagged cross-correlation, event correlation)
- [x] Phase 3 — RAG knowledge base: 7 SOPs + 9 past incidents, hybrid BM25 + embeddings with RRF
- [ ] Phase 4 — Agents + LangGraph orchestration
- [ ] Phase 5 — Guardrails + critic loop
- [ ] Phase 6 — Evaluation harness
- [ ] Phase 7 — MCP server, API, Docker

## Quick start
```bash
pip install -r requirements.txt
python -m rca.simulation      # creates data/plant.db and eval/ground_truth.json
pytest -q                     # 19 tests, no API key needed
```

Optional: `export OPENAI_API_KEY=...` switches retrieval from BM25-only to hybrid (BM25 + embeddings).

## The simulated plant
30 days of one paper machine (5-minute readings, 4-hour batches, 180 batches) with six injected faults:

| # | Fault | Why it is in the test set |
|---|---|---|
| F1 | Peroxide pump diaphragm tears | Cause → effect with a 30-min lag, plus a red-herring maintenance event |
| F2 | Steam valve leak lowers bleach temperature | Same symptom as F1, different cause |
| F3 | pH probe drifts high | Alarms fire but quality is fine — a sensor fault, not a process fault |
| F4 | Press felt left past its planned life | Gradual degradation, not a step change |
| F5 | New pulp supplier lot | No sensor deviates — only event correlation finds it |
| F6 | Disc filter screens blind | Environmental (effluent TSS) failure |

The true root causes live in `eval/ground_truth.json`, never in the database, so the agent cannot
look up the answers.

![F1 incident](docs/f1_peroxide_vs_brightness.png)
