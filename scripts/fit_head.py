"""Fit the readout that turns five left-or-right answers into a bearing (point.HEADS).

    python scripts/fit_head.py --upstream URL --camera windshield --cone-spacing 2.5

Frames come from wobbly drives - oracle-point with 10 deg of noise on every
answer, at 3 m/s, so the car is often off the line and angled into bends, as the
model's own driving leaves it - one frame every 3.6 s. Courses 0-2 fit the
weights; courses 10-12 check them. Each frame is asked the lane question in the
five turned views, each with its mirror image (half the difference cancels any
lean toward a side or a word), and the bearing is fitted as a linear function
of the five answers by least squares. Prints the weights and the error on both
sets; paste the weights into point.HEADS.
"""

from __future__ import annotations

import argparse, math, os, sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from PIL import ImageOps

from revdrive import point as P, render as R, sim as S
from revdrive.runner import observe

FIT, CHECK = (0, 1, 2), (10, 11, 12)


def frames(seeds, camera: str, spacing: float | None):
    """(seed, speed, aim distance, true bearing, five views) along wobbly oracle drives."""
    P.TOP = 3.0
    out, n = [], 0
    for seed in seeds:
        course = S.make_course(seed, spacing=spacing)
        sim, r = S.Sim(course, max_time=200), R.Renderer(course)
        oracle, tracker = P.OraclePoint(noise_deg=10, seed=seed, lead=0.6), P.Tracker()
        k = 0
        while not sim.done:
            if k % 30 == 0:
                a = oracle.act(observe(sim))
                tracker.set(a["target"], a["speed"])
            if k % 60 == 30 and sim.car.v > 1:
                if n % 3 == 0:
                    d = P.aim_distance(sim.car.v) + sim.car.v * 0.6
                    views = [r.render(SimpleNamespace(car=SimpleNamespace(x=sim.car.x, y=sim.car.y, yaw=sim.car.yaw + a),
                                                      cone_pos=sim.cone_pos, cone_down=sim.cone_down), 512, 288,
                                      R.VIEWS[camera]) for a in P.VIEWS]
                    out.append((seed, sim.car.v, d, P.true_bearing(course, sim.car, sim.idx, d), views))
                n += 1
            tracker.control(sim)
            sim.step()
            k += 1
        r.release()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", default=os.environ.get("REV_UPSTREAM"), required=not os.environ.get("REV_UPSTREAM"))
    p.add_argument("--camera", default="windshield", choices=sorted(R.VIEWS))
    p.add_argument("--cone-spacing", type=float, default=2.5, help="0 for sparse cones")
    p.add_argument("--workers", type=int, default=6, help="requests in flight; the server may be shared")
    a = p.parse_args()
    from rev.remote import Remote
    rows = frames(FIT + CHECK, a.camera, a.cone_spacing or None)
    print(f"{len(rows)} frames", file=sys.stderr)
    engines = [Remote(a.upstream) for _ in range(a.workers)]

    def ask(job):
        i, k, mirrored = job
        seed, v, d, _, views = rows[i]
        img = ImageOps.mirror(views[k]) if mirrored else views[k]
        state = [{"type": "text", "text": f"Speed: {v * 3.6:.0f} km/h."}, {"type": "image_url", "image_url": {"url": P.jpeg_url(img)}}]
        pr, _ = engines[(i * 10 + 2 * k + mirrored) % len(engines)].read_tokens(
            state, P.TASK + P.LANE_Q.format(d=d), ["left", "right"], "Answer with one word: left or right.")
        return i, k, mirrored, pr["left"] - pr["right"]

    jobs = [(i, k, m) for i in range(len(rows)) for k in range(len(P.VIEWS)) for m in (0, 1)]
    A = np.zeros((len(rows), len(P.VIEWS), 2))
    with ThreadPoolExecutor(a.workers) as ex:
        for i, k, m, val in ex.map(ask, jobs):
            A[i, k, m] = val
    X = (A[:, :, 0] - A[:, :, 1]) / 2
    y = np.degrees([r[3] for r in rows])
    seed = np.array([r[0] for r in rows])
    fit, check = np.isin(seed, FIT), np.isin(seed, CHECK)
    Z = np.column_stack([X, np.ones(len(y))])
    w = np.linalg.lstsq(Z[fit], y[fit], rcond=None)[0]
    err = np.abs(Z @ w - y)
    for name, m in (("fit (courses 0-2)", fit), ("check (courses 10-12)", check)):
        sharp = m & (np.abs(y) > 15)
        print(f"{name:22} n {m.sum():3}  mean error {err[m].mean():.1f} deg  median {np.median(err[m]):.1f}  "
              f"within 6 deg {np.mean(err[m] < 6):.0%}  bends past 15 deg {err[sharp].mean():.1f}  "
              f"(straight ahead would err {np.abs(y[m]).mean():.1f})")
    print(f'"{a.camera}": (np.array({[round(float(x), 1) for x in w[:-1]]}), {w[-1]:.1f}),')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
