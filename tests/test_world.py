"""The world and the harness, without a model: courses, rules, the oracle's ceiling."""
import math
from types import SimpleNamespace

import numpy as np

from revdrive import point as P, sim as S
from revdrive.runner import run

fail = 0
def check(cond, what):
    global fail
    print(("  ok    " if cond else "  FAIL  ") + what)
    fail += not cond

# Courses are deterministic, and clear of themselves.
a, b = S.make_course(3), S.make_course(3)
check(np.array_equal(a.centre, b.centre) and np.array_equal(a.cones, b.cones), "course 3 is the same twice")
check(not np.array_equal(S.make_course(3).centre, S.make_course(4).centre), "courses 3 and 4 differ")
c = S.make_course(S.DRIVINGBENCH, spacing=2.5)
check(125 < c.length < 140 and c.half_width == 4.0, f"course 100 is about 130 m and 8 m wide ({c.length:.0f} m)")
gaps = np.hypot(*np.diff(c.cones[c.cone_side == 1], axis=0).T)
check(abs(np.median(gaps) - 2.5) < 0.3, f"dense cones sit 2.5 m apart ({np.median(gaps):.2f})")
m = S.make_course(3, mirrored=True)
check(np.allclose(m.centre[:, 1], -a.centre[:, 1]), "a mirrored course is the mirror image")

# The strict rule: the first cone touched ends the run.
s = S.Sim(S.make_course(0), strict=True)
s.control(0.0, 3.0)
s.car.yaw += 0.3                                   # aim at the left line
while not s.done and s.t < 30:
    s.step()
check(s.status in ("hit a cone", "off course") and s.hits <= 1, f"strict: ends at the first cone ({s.status})")
s = S.Sim(S.make_course(0), strict=False)
s.control(0.0, 3.0); s.car.yaw += 0.3
while not s.done and s.t < 30:
    s.step()
check(s.status == "off course", f"lenient: runs on to 4 m off ({s.status}, {s.hits} cones)")

# The true bearing of a straight ahead is zero; the oracle through the controller finishes.
s = S.Sim(S.make_course(0))
check(abs(P.true_bearing(s.course, s.car, s.idx, 10.0)) < 1e-6, "on the line at the start, the course is dead ahead")
P.TOP = 1.5
for course in (S.DRIVINGBENCH, 0):
    summary, full = run(S.make_course(course, spacing=2.5), P.OraclePoint(lead=1.125), latency=1.0)
    check(summary["status"] == "finished" and summary["cones_hit"] == 0,
          f"oracle-point finishes course {course} at 1 s latency, strict ({summary['status']}, {summary['time']} s)")
    check(len(full["trace"]) > 100 and all("truth" in d for d in full["log"]), "the record has the trace and graded decisions")

print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
