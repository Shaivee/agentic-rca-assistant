"""
Tests for the RCA toolkit, run against the simulated plant.

Because we injected the faults ourselves, we know what each tool SHOULD find.
These tests prove the maths works before any LLM gets involved.

Run:  python -m rca.simulation && pytest -q
"""
import pytest

from rca import analytics as A
from rca.simulation import DB_PATH


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.skip("Run `python -m rca.simulation` first to create data/plant.db")
    c = A.connect(DB_PATH)
    yield c
    c.close()


def test_connection_is_read_only(con):
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM events")


def test_f1_anomaly_points_at_peroxide(con):
    top = A.detect_anomalies(con, "B-1034")
    flagged = [r["tag"] for r in top if r["flagged"]]
    assert "peroxide_dose" in flagged
    assert "brightness_online" in flagged


def test_f1_peroxide_leads_brightness_by_30_min(con):
    r = A.cross_correlate(con, "peroxide_dose", "brightness_online", "2026-09-06 06:00", "2026-09-06 21:00")
    assert r["best_lag_minutes"] == 30
    assert r["correlation_at_best_lag"] > 0.8


def test_f1_peroxide_is_top_driver(con):
    drivers = A.find_drivers(con, "brightness_online", "2026-09-06 06:00", "2026-09-06 21:00")
    assert drivers[0]["cause_tag"] == "peroxide_dose"


def test_f2_temperature_is_top_driver(con):
    drivers = A.find_drivers(con, "brightness_online", "2026-09-10 20:00", "2026-09-11 12:00")
    assert drivers[0]["cause_tag"] == "bleach_temp"


def test_f3_ph_alarm_without_quality_impact(con):
    flagged = {r["tag"] for r in A.detect_anomalies(con, "B-1087") if r["flagged"]}
    assert "bleach_ph" in flagged
    assert "brightness_online" not in flagged
    assert A.batch_context(con, "B-1087")["result"] == "PASS"


def test_f4_defects_differ_from_good_batches(con):
    cmp = A.compare_batches(con, "B-1120")
    assert cmp["quality"][0]["variable"] == "defects_per_km"


def test_f5_new_supplier_and_no_process_deviation(con):
    ctx = A.batch_context(con, "B-1147")
    assert ctx["new_supplier"] and ctx["supplier"] == "NorthFiber"
    flagged = {r["tag"] for r in A.detect_anomalies(con, "B-1147") if r["flagged"]}
    assert flagged <= {"brightness_online"}          # nothing upstream deviated
    start, end = A.batch_window(con, "B-1147")
    types = {e["event_type"] for e in A.correlate_events(con, start, end, event_types=["material_change"])}
    assert "material_change" in types


def test_events_link_equipment_to_tags(con):
    events = A.correlate_events(con, "2026-09-06 08:00", "2026-09-06 16:00", event_types=["maintenance"])
    p301 = [e for e in events if e["equipment"] == "P-301"]
    assert p301 and "peroxide_dose" in p301[0]["linked_tags"]
