"""
RCA analytics toolkit.

These are plain, deterministic Python functions - no LLM involved. The agent
(Phase 4) decides WHEN to call them; the maths itself is testable and repeatable.

Tools
-----
batch_context      What batch is this, did it pass, which pulp lot / supplier?
detect_anomalies   Which sensors behaved abnormally during the batch? (robust z-score)
compare_batches    How does this batch differ from recent GOOD batches? (effect size)
cross_correlate    Does tag A lead tag B, and by how many minutes? (lagged correlation)
find_drivers       Run cross_correlate for every candidate cause and rank them
correlate_events   Which maintenance logs / notes / alarms / material changes happened nearby?

Every function returns plain dicts/lists so results can be passed straight to an LLM as JSON.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "plant.db"
STEP_MINUTES = 5
MAD_SCALE = 1.4826   # MAD x 1.4826 ~ standard deviation, if the data were normal

# Tags that are effects (quality signals), not candidate causes.
QUALITY_TAGS = {"brightness_online", "defect_rate", "effluent_tss"}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the plant database READ-ONLY. Even if a bad query reaches this
    connection, SQLite itself will refuse to modify anything."""
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def batch_window(con: sqlite3.Connection, batch_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    row = con.execute("SELECT start_ts, end_ts FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown batch_id: {batch_id}")
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])


def get_series(con: sqlite3.Connection, tag: str, start, end) -> pd.Series:
    """Readings for one tag in [start, end)."""
    df = pd.read_sql(
        "SELECT ts, value FROM process_readings WHERE tag = ? AND ts >= ? AND ts < ? ORDER BY ts",
        con, params=(tag, str(pd.Timestamp(start)), str(pd.Timestamp(end))), parse_dates=["ts"],
    )
    return df.set_index("ts")["value"]


def _tags(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM tags", con).set_index("tag")


def _r(x: float, n: int = 3) -> float:
    return float(round(float(x), n))


# ---------------------------------------------------------------------------
# Tool 1: batch context
# ---------------------------------------------------------------------------
def batch_context(con: sqlite3.Connection, batch_id: str, new_supplier_days: int = 7) -> dict:
    """Basic facts about a batch: time window, quality result, pulp lot, and
    whether its pulp supplier is NEW (first used less than `new_supplier_days` ago).

    We ask "when was this supplier first used?" rather than "was it used in the
    previous batch?" - the second question would say a supplier is no longer new
    after a single batch, hiding exactly the kind of change that causes incidents.
    """
    b = pd.read_sql("SELECT * FROM batches WHERE batch_id = ?", con, params=(batch_id,))
    if b.empty:
        raise ValueError(f"Unknown batch_id: {batch_id}")
    b = b.iloc[0]
    q = pd.read_sql("SELECT * FROM quality_results WHERE batch_id = ?", con, params=(batch_id,)).iloc[0]
    lot = pd.read_sql("SELECT * FROM pulp_lots WHERE lot_id = ?", con, params=(b.pulp_lot,)).iloc[0]

    first_use = con.execute(
        """SELECT MIN(bt.start_ts) FROM batches bt JOIN pulp_lots l ON bt.pulp_lot = l.lot_id
           WHERE l.supplier = ?""", (lot.supplier,)).fetchone()[0]
    days_since_first_use = (pd.Timestamp(b.start_ts) - pd.Timestamp(first_use)) / pd.Timedelta(days=1)
    has_history = con.execute("SELECT COUNT(*) FROM batches WHERE start_ts < ?", (first_use,)).fetchone()[0] > 0
    return {
        "batch_id": batch_id,
        "start_ts": str(b.start_ts), "end_ts": str(b.end_ts),
        "result": q.result,
        "failed_specs": q.failed_specs.split(",") if q.failed_specs else [],
        "quality": {k: float(q[k]) for k in ["brightness_iso", "defects_per_km", "effluent_tss", "tensile_index"]},
        "pulp_lot": b.pulp_lot, "supplier": lot.supplier,
        "supplier_first_used": str(first_use),
        "new_supplier": bool(has_history and days_since_first_use < new_supplier_days),
    }


# ---------------------------------------------------------------------------
# Tool 2: anomaly detection (robust z-score)
# ---------------------------------------------------------------------------
def detect_anomalies(con: sqlite3.Connection, batch_id: str, baseline_days: int = 7,
                     z_threshold: float = 3.5, tags: list[str] | None = None) -> list[dict]:
    """Compare every tag during the batch with its own recent history.

    Why a ROBUST z-score?  A normal z-score uses mean and std, and both are
    pulled around by the very outliers we are trying to find (and by earlier
    incidents sitting in the baseline). The median and MAD (median absolute
    deviation) ignore extreme values, so the baseline stays honest.

        robust_z = (x - median(baseline)) / (1.4826 * MAD(baseline))

    A tag is flagged when its typical (median) z in the batch exceeds the
    threshold, or when at least 25% of its readings do.
    """
    start, end = batch_window(con, batch_id)
    meta = _tags(con)
    results = []
    for tag in tags or list(meta.index):
        base = get_series(con, tag, start - pd.Timedelta(days=baseline_days), start)
        cur = get_series(con, tag, start, end)
        if base.empty or cur.empty:
            continue
        med = base.median()
        mad = MAD_SCALE * (base - med).abs().median()
        mad = max(mad, 1e-6)
        z = (cur - med) / mad
        anomalous = z.abs() > z_threshold
        median_z = z.median()
        lo, hi = meta.loc[tag, "low_limit"], meta.loc[tag, "high_limit"]
        results.append({
            "tag": tag,
            "description": meta.loc[tag, "description"],
            "unit": meta.loc[tag, "unit"],
            "baseline_median": _r(med),
            "batch_median": _r(cur.median()),
            "median_robust_z": _r(median_z, 2),
            "pct_readings_anomalous": _r(100 * anomalous.mean(), 1),
            "direction": "high" if median_z > 0 else "low",
            "first_anomalous_ts": str(z.index[anomalous.argmax()]) if anomalous.any() else None,
            "readings_outside_limits": int(((cur < lo) | (cur > hi)).sum()),
            "flagged": bool(abs(median_z) > z_threshold or anomalous.mean() >= 0.25),
        })
    return sorted(results, key=lambda r: abs(r["median_robust_z"]), reverse=True)


# ---------------------------------------------------------------------------
# Tool 3: batch comparison (target vs recent GOOD batches)
# ---------------------------------------------------------------------------
def compare_batches(con: sqlite3.Connection, batch_id: str, n_reference: int = 12) -> dict:
    """Compare a batch with the last `n_reference` batches that PASSED.

    For every tag we compute the batch mean, then an effect size:
        effect = (target_mean - mean(reference means)) / std(reference means)
    i.e. "how many normal batch-to-batch variations away is this batch?"
    Comparing against good batches (not all batches) mirrors how engineers
    work: "what is different about the bad one?"
    """
    start, _ = batch_window(con, batch_id)
    refs = pd.read_sql(
        """SELECT b.batch_id FROM batches b JOIN quality_results q USING (batch_id)
           WHERE q.result = 'PASS' AND b.start_ts < ? ORDER BY b.start_ts DESC LIMIT ?""",
        con, params=(str(start), n_reference))["batch_id"].tolist()
    if len(refs) < 3:
        raise ValueError("Not enough earlier PASS batches to compare against.")

    ids = refs + [batch_id]
    placeholders = ",".join("?" * len(ids))
    means = pd.read_sql(
        f"""SELECT b.batch_id, r.tag, AVG(r.value) AS mean_value
            FROM batches b JOIN process_readings r
              ON r.ts >= b.start_ts AND r.ts < b.end_ts
            WHERE b.batch_id IN ({placeholders})
            GROUP BY b.batch_id, r.tag""", con, params=ids
    ).pivot(index="batch_id", columns="tag", values="mean_value")
    quality = pd.read_sql(
        f"""SELECT batch_id, brightness_iso, defects_per_km, effluent_tss, tensile_index
            FROM quality_results WHERE batch_id IN ({placeholders})""", con, params=ids
    ).set_index("batch_id")

    def effects(table: pd.DataFrame) -> list[dict]:
        ref, tgt = table.loc[refs], table.loc[batch_id]
        out = []
        for col in table.columns:
            sd = max(ref[col].std(), 1e-6)
            eff = (tgt[col] - ref[col].mean()) / sd
            out.append({"variable": col, "target": _r(tgt[col]), "reference_mean": _r(ref[col].mean()),
                        "difference": _r(tgt[col] - ref[col].mean()), "effect_size": _r(eff, 2)})
        return sorted(out, key=lambda r: abs(r["effect_size"]), reverse=True)

    return {"batch_id": batch_id, "reference_batches": refs,
            "process": effects(means), "quality": effects(quality)}


# ---------------------------------------------------------------------------
# Tool 4: lagged cross-correlation
# ---------------------------------------------------------------------------
def cross_correlate(con: sqlite3.Connection, cause_tag: str, effect_tag: str, start, end,
                    max_lag_minutes: int = 120) -> dict:
    """Does `cause_tag` lead `effect_tag`, and by how much?

    We shift the effect series later in time by 0, 5, 10 ... max_lag minutes and
    compute the Pearson correlation at each shift. The lag with the strongest
    correlation is the estimated delay between cause and effect.

    Why it matters for RCA: a real cause must happen BEFORE its effect, with a
    plausible, consistent delay (here ~30 min, the bleach tower residence time).
    A variable that only correlates at lag 0, or with a negative lag, is more
    likely a fellow symptom than the cause.
    """
    x = get_series(con, cause_tag, start, end)
    y = get_series(con, effect_tag, start, end)
    df = pd.concat({"x": x, "y": y}, axis=1).dropna()
    if len(df) < 12:
        raise ValueError("Window too short for cross-correlation (need >= 1 hour of data).")
    xs = (df.x - df.x.mean()) / max(df.x.std(), 1e-9)
    ys = (df.y - df.y.mean()) / max(df.y.std(), 1e-9)

    max_steps = min(max_lag_minutes // STEP_MINUTES, len(df) // 2)
    curve = []
    for k in range(max_steps + 1):
        a, b = (xs.values, ys.values) if k == 0 else (xs.values[:-k], ys.values[k:])
        curve.append((k * STEP_MINUTES, float(np.corrcoef(a, b)[0, 1])))
    best_lag, best_r = max(curve, key=lambda c: abs(c[1]))
    return {
        "cause_tag": cause_tag, "effect_tag": effect_tag,
        "window": [str(pd.Timestamp(start)), str(pd.Timestamp(end))],
        "best_lag_minutes": best_lag,
        "correlation_at_best_lag": _r(best_r),
        "correlation_at_lag_0": _r(curve[0][1]),
        "relationship": "positive" if best_r > 0 else "negative",
        "strength": "strong" if abs(best_r) >= 0.7 else "moderate" if abs(best_r) >= 0.4 else "weak",
        "curve_every_15_min": [(lag, _r(r)) for lag, r in curve if lag % 15 == 0],
    }


def find_drivers(con: sqlite3.Connection, effect_tag: str, start, end,
                 max_lag_minutes: int = 120, top_n: int = 5) -> list[dict]:
    """Run cross_correlate for every non-quality tag against `effect_tag` and rank
    candidate causes by the strength of their lagged correlation."""
    candidates = [t for t in _tags(con).index if t not in QUALITY_TAGS and t != effect_tag]
    results = [cross_correlate(con, t, effect_tag, start, end, max_lag_minutes) for t in candidates]
    for r in results:
        r.pop("curve_every_15_min")
    return sorted(results, key=lambda r: abs(r["correlation_at_best_lag"]), reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Tool 5: event correlation
# ---------------------------------------------------------------------------
def correlate_events(con: sqlite3.Connection, start, end, lookback_hours: int = 24,
                     after_hours: int = 12, event_types: list[str] | None = None) -> list[dict]:
    """Events in [start - lookback, end + after].

    Each event is tagged with:
      hours_from_start  negative = happened before the window started
      linked_tags       sensors whose description mentions the event's equipment
                        (e.g. maintenance on P-301 links to peroxide_dose)
    Events AFTER the window matter too: a repair log written later often names
    the root cause ("diaphragm found torn").
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    ev = pd.read_sql(
        "SELECT * FROM events WHERE ts >= ? AND ts < ? ORDER BY ts", con,
        params=(str(start - pd.Timedelta(hours=lookback_hours)), str(end + pd.Timedelta(hours=after_hours))),
        parse_dates=["ts"],
    )
    if event_types:
        ev = ev[ev.event_type.isin(event_types)]
    meta = _tags(con)
    out = []
    for e in ev.itertuples():
        linked = [t for t, d in meta["description"].items() if e.equipment and e.equipment in d] \
            or ([e.equipment] if e.equipment in meta.index else [])
        out.append({"event_id": int(e.event_id), "ts": str(e.ts), "event_type": e.event_type,
                    "equipment": e.equipment, "description": e.description,
                    "hours_from_start": _r((e.ts - start) / pd.Timedelta(hours=1), 2),
                    "linked_tags": linked})
    return out
