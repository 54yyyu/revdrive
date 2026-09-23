"""The cameras: projection, determinism, and renderers sharing a thread."""
import math

import numpy as np

from revdrive import render as R, sim as S

fail = 0
def check(cond, what):
    global fail
    print(("  ok    " if cond else "  FAIL  ") + what)
    fail += not cond

course = S.make_course(0, spacing=2.5)
sim = S.Sim(course)
car = sim.car

# A point on the ground straight ahead projects onto the middle column, below the horizon.
ahead = np.array([[car.x + 20 * math.cos(car.yaw), car.y + 20 * math.sin(car.yaw), 0.0]])
for name, cam in R.CAMS.items():
    px = R.project(car, ahead, 512, 288, cam)[0]
    off = px[0] - 256 if name != "driver" else 0.0      # the driver sits left of the middle
    check(abs(off) < 1 and 144 < px[1] < 288, f"{name}: dead ahead lands mid-picture ({px[0]:.0f}, {px[1]:.0f})")
left = np.array([[car.x + 20 * math.cos(car.yaw + 0.3), car.y + 20 * math.sin(car.yaw + 0.3), 0.0]])
check(R.project(car, left, 512, 288, R.WINDSHIELD)[0][0] < 256, "a point to the left lands left")

# Frames are deterministic, and a second renderer in the same thread does not disturb the first.
a = R.Renderer(course)
one = np.asarray(a.render(sim, 512, 288, R.WINDSHIELD)).astype(int)
check(np.array_equal(one, np.asarray(a.render(sim, 512, 288, R.WINDSHIELD)).astype(int)), "the same frame twice")
b = R.Renderer(S.make_course(S.DRIVINGBENCH, spacing=2.5))
b.render(S.Sim(b.course), 960, 540, R.WIDE)
check(np.array_equal(one, np.asarray(a.render(sim, 512, 288, R.WINDSHIELD)).astype(int)),
      "a renderer draws the same after another is made in its thread")
a.release(); b.release()

print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
