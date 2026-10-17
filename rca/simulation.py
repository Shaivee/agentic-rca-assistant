"""
Synthetic paper-mill data with injected faults.

Why synthetic data?
  1. Real plant data is confidential.
  2. Because WE inject every fault, we know the true root cause of every
     incident. That ground truth is what lets us measure the agent's accuracy.

The ground truth is written to eval/ground_truth.json, NOT into the database.
The agent only sees what a real engineer would see (sensors, events, quality
results). Putting the answers in the DB would be label leakage.

Run:
    python -m rca.simulation
Outputs:
    data/plant.db             SQLite database the agent will query
    eval/ground_truth.json    the true root cause of each injected incident
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "plant.db"
GROUND_TRUTH_PATH = ROOT / "eval" / "ground_truth.json"

SEED = 42
START = pd.Timestamp("2026-09-01 00:00")
DAYS = 30
STEP = pd.Timedelta(minutes=5)       # one reading every 5 minutes
BATCH = pd.Timedelta(hours=4)        # one batch every 4 hours -> 180 batches
BRIGHTNESS_LAG = 6                   # 6 steps x 5 min = 30 min from bleaching to brightness sensor

# tag: (description, unit, baseline, noise_sd, low_limit, high_limit)
# baseline=None means the tag is computed from other tags (it is an *effect*, not a cause).
TAGS = {
    "peroxide_dose":     ("H2O2 dose to bleach tower (pump P-301)", "kg/t", 25.0, 0.6, 22.0, 28.0),
    "caustic_dose":      ("NaOH dose to bleach tower", "kg/t", 15.0, 0.4, 13.0, 17.0),
    "bleach_temp":       ("Bleach tower temperature (steam valve TV-210)", "degC", 75.0, 0.7, 70.0, 80.0),
    "bleach_ph":         ("Bleach tower pH (probe AT-305)", "pH", 10.8, 0.06, 10.4, 11.2),
    "stock_consistency": ("Stock consistency", "%", 12.0, 0.2, 11.0, 13.0),
    "machine_speed":     ("Paper machine speed", "m/min", 900.0, 6.0, 870.0, 930.0),
    "filter_dp":         ("Disc filter DF-1 differential pressure", "kPa", 30.0, 1.2, 20.0, 42.0),
    "brightness_online": ("Online brightness after bleaching", "%ISO", None, 0.25, 84.0, 90.0),
    "defect_rate":       ("Web Inspection System (WIS) defect rate", "defects/km", None, 0.25, 0.0, 3.5),
    "effluent_tss":      ("Effluent total suspended solids", "mg/L", None, 1.5, 0.0, 40.0),
}

# Batch-level quality specs. A batch fails if ANY spec is violated.
SPECS = {"brightness_iso": (84.0, None), "defects_per_km": (None, 3.5),
         "effluent_tss": (None, 40.0), "tensile_index": (60.0, None)}

rng = np.random.default_rng(SEED)
TIMES = pd.date_range(START, START + pd.Timedelta(days=DAYS), freq=STEP, inclusive="left")
N = len(TIMES)


def at(ts: str) -> int:
    """Index of a timestamp like '2026-09-06 09:00' in the TIMES array."""
    return int((pd.Timestamp(ts) - START) / STEP)


def ar1_noise(sd: float, phi: float = 0.9) -> np.ndarray:
    """Autocorrelated noise. Real sensors drift smoothly; they don't jump randomly
    every 5 minutes. AR(1): x[t] = phi * x[t-1] + e[t], scaled so std(x) = sd."""
    e = rng.normal(0, sd * np.sqrt(1 - phi**2), N)
    x = np.empty(N)
    x[0] = rng.normal(0, sd)
    for t in range(1, N):
        x[t] = phi * x[t - 1] + e[t]
    return x


def first_order_step(start: str, end: str, magnitude: float, tau: float = 6.0) -> np.ndarray:
    """An offset that moves towards `magnitude` exponentially (like a real process
    responding to a change), then decays back after `end`. tau is in steps."""
    off = np.zeros(N)
    s, e = at(start), at(end)
    k = np.arange(e - s)
    off[s:e] = magnitude * (1 - np.exp(-k / tau))
    tail = np.arange(N - e)
    off[e:] = off[e - 1] * np.exp(-tail / tau)
    return off


def lagged(x: np.ndarray, steps: int) -> np.ndarray:
    """x shifted later in time by `steps` (the effect appears after the cause)."""
    out = np.roll(x, steps)
    out[:steps] = x[0]
    return out


# ---------------------------------------------------------------------------
# 1. Normal operation: every input tag = baseline + autocorrelated noise
# ---------------------------------------------------------------------------
true = {tag: base + ar1_noise(sd) for tag, (_, _, base, sd, _, _) in TAGS.items() if base is not None}
measured_ph = true["bleach_ph"].copy()   # what the probe reports (differs from reality in F3)

# ---------------------------------------------------------------------------
# 2. Batches and pulp lots (lots normally change every 2 days, same supplier)
# ---------------------------------------------------------------------------
batch_starts = pd.date_range(START, START + pd.Timedelta(days=DAYS), freq=BATCH, inclusive="left")
F5_START, F5_END = pd.Timestamp("2026-09-25 04:00"), pd.Timestamp("2026-09-26 00:00")

batches, lots = [], {}
for k, bstart in enumerate(batch_starts):
    lot = "NF-0001" if F5_START <= bstart < F5_END else f"ES-{2601 + k // 12}"
    if lot not in lots:
        supplier = "NorthFiber" if lot.startswith("NF") else "Eucalipto SA"
        lots[lot] = {"lot_id": lot, "supplier": supplier,
                     "received_date": (bstart - pd.Timedelta(days=1)).date().isoformat(),
                     # hidden property of the pulp: how much it shifts brightness
                     "_brightness_effect": -2.6 if lot.startswith("NF") else rng.normal(0, 0.15)}
    batches.append({"batch_id": f"B-{1001 + k}", "line": "PM1", "grade": "Copy 80gsm",
                    "start_ts": bstart, "end_ts": bstart + BATCH, "pulp_lot": lot})

lot_effect = np.zeros(N)
for b in batches:
    lot_effect[at(str(b["start_ts"])):at(str(b["end_ts"]))] = lots[b["pulp_lot"]]["_brightness_effect"]

# ---------------------------------------------------------------------------
# 3. Inject the faults
# ---------------------------------------------------------------------------
# F1  Peroxide pump P-301 diaphragm tears: dose ramps down to 62%, fixed at 17:30.
factor = np.ones(N)
s, m, e = at("2026-09-06 09:00"), at("2026-09-06 12:00"), at("2026-09-06 17:30")
factor[s:m] = np.linspace(1.0, 0.62, m - s)
factor[m:e] = 0.62
true["peroxide_dose"] *= factor
true["peroxide_dose"][s:e] += rng.normal(0, 0.8, e - s)        # unsteady flow from the damaged pump

# F2  Steam valve TV-210 positioner leak: bleach temperature falls ~8 degC.
true["bleach_temp"] += first_order_step("2026-09-11 00:30", "2026-09-11 07:30", -8.0)

# F3  pH probe AT-305 fouls and reads HIGH. The real pH is unchanged -> a sensor fault, not a process fault.
s, m, e = at("2026-09-15 02:00"), at("2026-09-15 12:00"), at("2026-09-15 14:00")
measured_ph[s:m] += np.linspace(0, 0.7, m - s)
measured_ph[m:e] += 0.7

# F4  Press felt PF-2 is left in service past its 18-day planned life. Wear -> holes/spots.
felt_changes = [pd.Timestamp("2026-08-31 00:00"), pd.Timestamp("2026-09-22 08:00")]
felt_age_days = np.empty(N)
for i, t in enumerate(TIMES):
    last = max(c for c in felt_changes if c <= t)
    felt_age_days[i] = (t - last) / pd.Timedelta(days=1)

# F5  Pulp lot NF-0001 from a new supplier bleaches poorly (handled by lot_effect above).

# F6  Disc filter DF-1 screens blind with fines: differential pressure climbs.
s, e = at("2026-09-28 06:00"), at("2026-09-28 16:00")
true["filter_dp"][s:e] += np.linspace(0, 22, e - s)

# ---------------------------------------------------------------------------
# 4. Effects: quality signals are computed from the (true) causes
# ---------------------------------------------------------------------------
brightness = (86.2
              + 0.35 * (lagged(true["peroxide_dose"], BRIGHTNESS_LAG) - 25.0)
              + 0.30 * (lagged(true["bleach_temp"], BRIGHTNESS_LAG) - 75.0)
              + lot_effect
              + ar1_noise(0.25))
defect_rate = (1.8
               + 0.45 * np.clip(felt_age_days - 16, 0, None)        # felt wear kicks in after ~16 days
               + 0.01 * (true["machine_speed"] - 900)
               + ar1_noise(0.25))
defect_rate = np.clip(defect_rate, 0, None)
effluent_tss = 24 + 1.1 * (true["filter_dp"] - 30) + ar1_noise(1.5)

readings = dict(true)
readings["bleach_ph"] = measured_ph          # the database stores what the probe REPORTS
readings["brightness_online"] = brightness
readings["defect_rate"] = defect_rate
readings["effluent_tss"] = effluent_tss

# ---------------------------------------------------------------------------
# 5. Batch quality results (lab tests = batch average + small lab error)
# ---------------------------------------------------------------------------
quality = []
for b in batches:
    s, e = at(str(b["start_ts"])), at(str(b["end_ts"]))
    q = {"batch_id": b["batch_id"],
         "brightness_iso": round(brightness[s:e].mean() + rng.normal(0, 0.15), 2),
         "defects_per_km": round(defect_rate[s:e].mean(), 2),
         "effluent_tss": round(effluent_tss[s:e].mean(), 1),
         "tensile_index": round(65 + rng.normal(0, 1.2), 1)}
    failed = [k for k, (lo, hi) in SPECS.items()
              if (lo is not None and q[k] < lo) or (hi is not None and q[k] > hi)]
    q["result"] = "FAIL" if failed else "PASS"
    q["failed_specs"] = ",".join(failed)
    quality.append(q)

# ---------------------------------------------------------------------------
# 6. Events: maintenance logs, operator notes, material changes, alarms, and noise
# ---------------------------------------------------------------------------
events = [
    # F1 clues + a red herring at the same time
    ("2026-09-06 10:40", "operator_note", "P-301", "P-301 discharge pressure fluctuating, peroxide flow looks unsteady."),
    ("2026-09-06 11:05", "maintenance", "C-12", "Conveyor C-12 belt tracking adjusted."),
    ("2026-09-06 17:30", "maintenance", "P-301", "Peroxide dosing pump P-301: diaphragm found torn, replaced. Flow restored."),
    # F2
    ("2026-09-11 07:30", "maintenance", "TV-210", "Steam valve TV-210: positioner air supply leak repaired, valve recalibrated."),
    # F3
    ("2026-09-15 14:00", "maintenance", "AT-305", "pH probe AT-305: reference junction fouled, cleaned and recalibrated. "
                                                  "Probe was reading ~0.7 pH high; grab sample showed 10.8."),
    # F4
    ("2026-09-18 09:00", "operator_note", "WIS", "Slight increase in WIS defects, mostly small holes and spots."),
    ("2026-09-22 08:00", "maintenance", "PF-2", "Press felt PF-2 replaced. Felt was 22 days in service (planned life 18 days)."),
    # F5
    ("2026-09-25 04:00", "material_change", "PULP", "Pulp feed switched to lot NF-0001 (supplier: NorthFiber, first delivery)."),
    ("2026-09-26 00:00", "material_change", "PULP", "Pulp feed switched back to supplier Eucalipto SA."),
    # F6
    ("2026-09-28 16:00", "maintenance", "DF-1", "Disc filter DF-1: screens blinded with fines, high-pressure wash performed."),
]
# Routine noise events, so the relevant ones are not the only ones in the log.
routine = [("maintenance", "LUB", "Weekly lubrication round completed."),
           ("operator_note", "SHIFT", "Shift handover: no issues reported."),
           ("maintenance", "HVAC", "Control room HVAC filter replaced."),
           ("operator_note", "QC", "Routine lab calibration check passed.")]
for day in range(DAYS):
    for _ in range(rng.integers(1, 3)):
        etype, equip, desc = routine[rng.integers(len(routine))]
        t = START + pd.Timedelta(days=day, hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
        events.append((str(t), etype, equip, desc))

# Alarms: one event each time a tag first crosses a limit (not every 5-min reading).
for tag, (desc, unit, _, _, lo, hi) in TAGS.items():
    x = readings[tag]
    outside = (x < lo) | (x > hi)
    starts = np.flatnonzero(outside & ~np.roll(outside, 1))
    for i in starts:
        kind = "LOW" if x[i] < lo else "HIGH"
        events.append((str(TIMES[i]), "alarm", tag, f"{tag} {kind} alarm: {x[i]:.2f} {unit} (limits {lo}-{hi})."))

events_df = pd.DataFrame(events, columns=["ts", "event_type", "equipment", "description"])
events_df["ts"] = pd.to_datetime(events_df["ts"], format="ISO8601")
events_df = events_df.sort_values("ts").reset_index(drop=True)
events_df.insert(0, "event_id", range(1, len(events_df) + 1))

# ---------------------------------------------------------------------------
# 7. Ground truth (for evaluation only - never shown to the agent)
# ---------------------------------------------------------------------------
INCIDENTS = [
    dict(id="F1", window=("2026-09-06 09:00", "2026-09-06 18:30"), category="dosing_equipment",
         equipment="P-301", quality_impact=True, symptom="low brightness",
         root_cause="Peroxide dosing pump P-301 diaphragm tore, cutting H2O2 dose ~38%; brightness fell ~30 min later.",
         key_evidence=["peroxide_dose drop", "30 min lag peroxide->brightness", "P-301 maintenance log"],
         distractors=["Conveyor C-12 maintenance at 11:05"]),
    dict(id="F2", window=("2026-09-11 00:30", "2026-09-11 08:30"), category="steam_supply",
         equipment="TV-210", quality_impact=True, symptom="low brightness",
         root_cause="Steam valve TV-210 positioner leak dropped bleach temperature ~8 degC, reducing bleaching.",
         key_evidence=["bleach_temp drop", "TV-210 maintenance log"], distractors=[]),
    dict(id="F3", window=("2026-09-15 02:00", "2026-09-15 14:00"), category="sensor_fault",
         equipment="AT-305", quality_impact=False, symptom="high pH alarms",
         root_cause="pH probe AT-305 fouled and read ~0.7 high. Real pH was normal; product quality unaffected.",
         key_evidence=["bleach_ph alarms", "brightness normal", "AT-305 recalibration log"],
         distractors=["pH alarms suggest a process problem that does not exist"]),
    dict(id="F4", window=("2026-09-18 00:00", "2026-09-22 08:00"), category="equipment_wear",
         equipment="PF-2", quality_impact=True, symptom="high defect rate",
         root_cause="Press felt PF-2 left in service past its 18-day planned life; wear raised WIS defects until replaced.",
         key_evidence=["gradual defect_rate rise", "felt age 22 days", "PF-2 replacement log"], distractors=[]),
    dict(id="F5", window=("2026-09-25 04:00", "2026-09-26 00:00"), category="raw_material",
         equipment="PULP", quality_impact=True, symptom="low brightness",
         root_cause="Pulp lot NF-0001 from new supplier NorthFiber bleached poorly; no process variable deviated.",
         key_evidence=["no process deviation", "material change to NF-0001", "brightness drop matches lot window"],
         distractors=[]),
    dict(id="F6", window=("2026-09-28 06:00", "2026-09-28 17:00"), category="filtration",
         equipment="DF-1", quality_impact=True, symptom="high effluent TSS",
         root_cause="Disc filter DF-1 screens blinded with fines; rising differential pressure let solids through.",
         key_evidence=["filter_dp rise", "effluent_tss rise", "DF-1 wash log"], distractors=[]),
]
for inc in INCIDENTS:
    w0, w1 = (pd.Timestamp(x) for x in inc["window"])
    overlapping = [b for b in batches if b["start_ts"] < w1 and b["end_ts"] > w0]
    inc["affected_batches"] = [b["batch_id"] for b in overlapping]
    inc["failed_batches"] = [q["batch_id"] for q in quality
                             if q["batch_id"] in inc["affected_batches"] and q["result"] == "FAIL"]

# ---------------------------------------------------------------------------
# 8. Write everything
# ---------------------------------------------------------------------------
def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    tags_df = pd.DataFrame([{"tag": t, "description": d, "unit": u, "low_limit": lo, "high_limit": hi}
                            for t, (d, u, _, _, lo, hi) in TAGS.items()])
    readings_df = pd.DataFrame({"ts": np.repeat(TIMES, len(readings)),
                                "tag": np.tile(list(readings), N),
                                "value": np.column_stack(list(readings.values())).ravel().round(3)})
    lots_df = pd.DataFrame([{k: v for k, v in lot.items() if not k.startswith("_")} for lot in lots.values()])
    with sqlite3.connect(DB_PATH) as con:
        tags_df.to_sql("tags", con, index=False)
        readings_df.to_sql("process_readings", con, index=False)
        pd.DataFrame(batches).to_sql("batches", con, index=False)
        lots_df.to_sql("pulp_lots", con, index=False)
        pd.DataFrame(quality).to_sql("quality_results", con, index=False)
        events_df.to_sql("events", con, index=False)
        con.execute("CREATE INDEX idx_readings ON process_readings(tag, ts)")
        con.execute("CREATE INDEX idx_events_ts ON events(ts)")

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.write_text(json.dumps(INCIDENTS, indent=2))

    n_fail = sum(q["result"] == "FAIL" for q in quality)
    print(f"Wrote {DB_PATH.relative_to(ROOT)}: {len(readings_df):,} readings, {len(batches)} batches "
          f"({n_fail} failed), {len(events_df)} events")
    for inc in INCIDENTS:
        print(f"  {inc['id']} {inc['category']:<16} affected={inc['affected_batches']} failed={inc['failed_batches']}")


if __name__ == "__main__":
    main()
