"""Saved runs: the record round-trips and replays the moment it was recorded."""
import tempfile
from pathlib import Path

import numpy as np

from revdrive import point as P, render as R, sim as S
from revdrive.runner import key, run
from revdrive.web import Store, state_at

fail = 0
def check(cond, what):
    global fail
    print(("  ok    " if cond else "  FAIL  ") + what)
    fail += not cond

P.TOP = 1.5
course = S.make_course(S.DRIVINGBENCH, spacing=2.5)
summary, full = run(course, P.OraclePoint(noise_deg=12, lead=1.125), latency=1.0)
with tempfile.TemporaryDirectory() as d:
    store = Store(Path(d) / "runs.jsonl")
    store.save(summary, full)
    back = store.load(summary["id"])
    check(back is not None and back["trace"] == full["trace"] and back["log"] == full["log"], "a record round-trips")
    check([s["id"] for s in store.summaries()] == [summary["id"]], "the summary is listed")
    check(key(summary) == key(back), "a rerun of the same experiment has the same key")

# Replaying a recorded moment gives the car where the trace had it.
t = full["trace"][len(full["trace"]) // 2]
st = state_at(full, course, S.Sim(course).cone_pos.copy(), t[0])
check(abs(st.car.x - t[1]) < 1e-6 and abs(st.car.y - t[2]) < 1e-6, "state_at returns the recorded pose")
check(int(st.cone_down.sum()) == sum(1 for e in full["cones"] if e[0] <= t[0]), "and the cones down by then")

print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
