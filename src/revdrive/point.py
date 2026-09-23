"""Point, don't steer: the model says where the course goes; a controller drives there.

A one-pass decision model gives a probability over a few answer tokens. Asked
for a steering level it leans on whichever option comes first, and asked "how
should you turn the wheel" it steers away from the cones in front of it - in a
bend, the inside line. What it answers well is a comparison: is the middle of
the lane left or right of the middle of this picture? So the model only ever
says *where the course is*, as a bearing from the car's heading, and driving
belongs to the controller:

- The bearing becomes a point on the ground, fixed in the world at the pose the
  picture was taken from. Between answers, pure pursuit on odometry steers to it,
  so a late answer is still an answer about the right place. The point sits
  further out by the distance covered while the answer is on its way (the
  latency is measured from the model's own answers).
- The model picks its own speed, as a digit, under a ceiling.

How each piece was chosen, and what was tried and dropped, is in DECISIONS.md.
"""

from __future__ import annotations

import base64, io, math, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import ImageDraw, ImageOps

from . import render as R
from . import sim as S

TASK = ("You are driving this car through an autocross course in an empty parking lot. "
        "Picture 1 is your view through the windshield. The course runs between two lines of "
        "orange cones, one line on each side; the car must stay between them.")
LANE_Q = (" Look at the middle of the course about {d:.0f} m ahead, halfway between the two lines of cones. "
          "Is it to the left or to the right of the middle of the picture?")
SPEED_Q = (" How fast should you drive right now? Answer with one digit: 0 means stop, 9 means as fast as "
           "you are allowed ({top:.1f} m/s); go slower where the course turns or where you are unsure of it.")

TOP = 1.5            # m/s, the ceiling; set from --top-speed
FAR = 2.0            # the oracle's speed point sits this many times further out than its steering point
REACH = 5.0          # m: at most this far travelled blind while an answer is on its way

# The readout: the lane question asked of five views turned to these headings,
# each read with its mirror image (half the difference cancels any lean toward
# one side or one word); the bearing is a linear function of the five answers.
# The weights are fitted by scripts/fit_head.py on frames of wobbly drives over
# courses 0-2 and checked on courses 10-12, per camera: "windshield" (DrivingBench's
# mount, 90 deg views from the wide camera, a cone every 2.5 m) errs 3.9 deg
# (median 3.2); "driver" (the left seat, 90 deg, our sparser cones) 4.4 (3.7).
VIEWS = np.radians([30, 15, 0, -15, -30])
HEADS = {"windshield": (np.array([34.4, 20.1, -1.8, 21.3, 28.1]), 1.4),
         "driver": (np.array([38.4, 36.9, 10.4, 28.6, 39.3]), 1.5)}


def aim_distance(v: float) -> float:
    """How far ahead the steering point sits, before the latency lead: from the
    nearest the ground shows past the hood (about 7 m) outward with speed."""
    return float(np.clip(5.0 + 0.6 * v, 8.0, 14.0))


def true_bearing(course: S.Course, car, idx: int, d: float) -> float:
    """Bearing of the centreline where it is `d` metres from the car, ahead of index `idx`."""
    c = course.centre
    ahead = c[idx: idx + int(4 * d / S.STEP)]
    dist = np.hypot(ahead[:, 0] - car.x, ahead[:, 1] - car.y)
    far = np.flatnonzero(dist >= d)
    p = ahead[far[0]] if len(far) else ahead[-1]
    a = math.atan2(p[1] - car.y, p[0] - car.x) - car.yaw
    return (a + math.pi) % (2 * math.pi) - math.pi


def target_speed(bearings, ds, lead: float = 0.0, lat_accel: float = 4.0, top: float | None = None) -> float:
    """The oracle's speed: as fast as the sharpest turn among the points allows,
    and slower when answers are slow, so the car never covers more than REACH
    metres blind."""
    top = TOP if top is None else top
    k = max(abs(2 * math.sin(b) / d) for b, d in zip(bearings, ds))
    v = min(top, math.sqrt(lat_accel / k) if k > 1e-6 else top)
    if lead > 0:
        v = min(v, REACH / lead)
    return max(min(3.0, top), v)


def _target(o, bearing: float, d: float) -> list[float]:
    a = o.yaw + bearing
    return [o.x + d * math.cos(a), o.y + d * math.sin(a)]


class Tracker:
    """Pure pursuit to a fixed point on the ground, and a speed loop."""

    def __init__(self, smooth: float = 0.25):
        self.target: tuple[float, float] | None = None
        self.speed = 0.0
        self.smooth = smooth

    def set(self, target, speed) -> None:
        """A new answer. Successive answers describe the same stretch of road, so
        the target moves most of the way toward it rather than jumping: their
        errors are independent and partly cancel."""
        t = np.array(target, float)
        if self.target is not None and self.smooth and np.hypot(*(t - self.target)) < 6.0:
            t = self.smooth * np.array(self.target) + (1 - self.smooth) * t
        self.target, self.speed = (float(t[0]), float(t[1])), float(speed)

    def control(self, sim: S.Sim) -> None:
        c = sim.car
        steer = c.wheel / S.MAX_WHEEL                    # no target, or passed it: hold the wheel
        if self.target is not None:
            dx, dy = self.target[0] - c.x, self.target[1] - c.y
            ahead = dx * math.cos(c.yaw) + dy * math.sin(c.yaw)
            if ahead > 2.0:
                alpha = math.atan2(dy, dx) - c.yaw
                steer = math.atan(2 * S.WHEELBASE * math.sin(alpha) / math.hypot(dx, dy)) / S.MAX_WHEEL
        accel = float(np.clip(1.5 * (self.speed - c.v), -S.MAX_BRAKE, S.MAX_ACCEL))
        sim.control(steer, accel)


class OraclePoint:
    """The true bearing of the centreline through the same controller: what the
    harness allows. `noise_deg` adds that much error to every answer, to see how
    good a model's bearing has to be."""
    needs_frame = False

    def __init__(self, noise_deg: float = 0.0, seed: int = 0, lead: float = 0.0):
        self.noise, self.lead = math.radians(noise_deg), lead
        self.rng = np.random.default_rng(seed)
        self.name = "oracle-point" if not noise_deg else f"oracle-point~{noise_deg:g}"

    def act(self, o, frames=None) -> dict:
        d = aim_distance(o.v) + o.v * self.lead
        b, bf = (true_bearing(o.course, o, o.idx, x) + (self.rng.normal(0, self.noise) if self.noise else 0.0)
                 for x in (d, FAR * d))
        return {"bearing": round(b, 4), "far": round(bf, 4), "d": round(d, 2), "target": _target(o, b, d),
                "speed": target_speed((b, bf), (d, FAR * d), self.lead)}


def jpeg_url(im, quality: int = 85) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Pointer:
    """rev, reading a served vision model: five views and their mirrors for the
    bearing, one question for the speed, all asked at once."""
    needs_frame = True

    def __init__(self, upstream: str, head: str = "windshield", blind: bool = False, workers: int = 6):
        from rev.remote import Remote
        self.head, self.blind = head, blind
        self.views = list(VIEWS)
        # One engine per worker: each keeps its own tokenizer, which is not safe to share.
        self.engines = [Remote(upstream) for _ in range(workers)]
        self.pool = ThreadPoolExecutor(workers)
        self.name = "blind" if blind else "rev"
        self.lead = 0.4            # seconds from picture to answer, learned from the answers

    def _read(self, k, state, question, answers, system):
        engine = self.engines[k % len(self.engines)]

        def read():
            # A shared server drops a connection now and then; retry briefly before
            # giving up (the car keeps moving meanwhile, and the delay is charged).
            for attempt in range(3):
                try:
                    return engine.read_tokens(state, question, answers, system)
                except Exception:                           # noqa: BLE001 - raised again on the last try
                    if attempt == 2:
                        raise
                    time.sleep(1.0 + attempt)
        return self.pool.submit(read)

    def act(self, o, frames) -> dict:
        if self.blind:
            frames = [f.point(lambda _: 128) for f in frames]
        t0 = time.perf_counter()
        d = aim_distance(o.v) + o.v * self.lead
        text = f"Speed: {o.v * 3.6:.0f} km/h."
        state = lambda im: [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": jpeg_url(im)}}]
        speed_f = self._read(len(frames) * 2, state(frames[len(frames) // 2]), TASK + SPEED_Q.format(top=TOP),
                             [str(i) for i in range(10)], "Answer with a single digit.")
        lane = [self._read(2 * k + j, state(img), TASK + LANE_Q.format(d=d), ["left", "right"],
                           "Answer with one word: left or right.")
                for k, im in enumerate(frames) for j, img in enumerate((im, ImageOps.mirror(im)))]
        got = [f.result()[0] for f in lane]
        # + : the lane's middle is left of that view's middle
        vals = np.array([((got[2 * k]["left"] - got[2 * k]["right"]) - (got[2 * k + 1]["left"] - got[2 * k + 1]["right"])) / 2
                         for k in range(len(frames))])
        w, b0 = HEADS[self.head]
        b = math.radians(float(np.clip(vals @ w + b0, -45, 45)))
        ps = np.array([speed_f.result()[0][str(i)] for i in range(10)])
        speed = float(ps @ np.arange(10)) / 9 * TOP     # the model's own choice; the controller holds it
        self.lead = 0.8 * self.lead + 0.2 * (time.perf_counter() - t0)
        return {"views": [round(v, 4) for v in vals], "speed_p": [round(v, 4) for v in ps],
                "bearing": round(b, 4), "d": round(d, 2), "target": _target(o, b, d), "speed": speed,
                "conf": round(float(np.clip(np.abs(vals).mean() * 4, 0, 1)), 4),
                "seconds": round(time.perf_counter() - t0, 3)}


def draw_target(frame, car, target, cam=R.DRIVER):
    """A ring on the ground where the controller is headed: for people watching, never for the model."""
    w, h = frame.size
    ring = [(target[0] + 0.6 * math.cos(a), target[1] + 0.6 * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 25)]
    px = R.project(car, np.array(ring), w, h, cam)
    if not np.isfinite(px).all():
        return frame
    im = frame.copy()
    ImageDraw.Draw(im).line([tuple(p) for p in px], fill=(59, 91, 219), width=max(2, w // 320))
    return im
