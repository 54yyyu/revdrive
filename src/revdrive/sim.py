"""The world: a cone course in a parking lot, and a car that drives on it.

Coordinates are metres, x east, y north, z up; yaw is counter-clockwise from +x.
The car's own frame puts x forward, y to the left, origin on the rear axle.

The course is what an autocross lays out: a centreline of straights and arcs,
a line of cones on each side, a checkered start and finish. It is scored the
way DrivingBench scores its cone course: progress is how far along the
centreline the car has got while within `OFF_COURSE` metres of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HALF_WIDTH = 3.0        # cones sit this far either side of the centreline
OFF_COURSE = 4.0        # further than this from the centreline ends the run
STEP = 0.5              # centreline sample spacing

# A Toyota Corolla (E210): wheelbase, overhangs, width; about 5.2 m kerb-to-kerb
# turning radius, so road-wheel lock is atan(2.70 / 5.2).
WHEELBASE = 2.70
FRONT_OVERHANG, REAR_OVERHANG = 0.93, 1.00
CAR_WIDTH = 1.78
MAX_WHEEL = math.atan(WHEELBASE / 5.2)
STEER_TIME = 0.6        # seconds from centre to full lock
PEDAL_LAG = 0.25        # first-order lag on longitudinal acceleration
GRIP = 8.5              # m/s^2 of lateral acceleration before the front washes out
MAX_ACCEL, MAX_BRAKE = 3.0, 8.0
CONE_R = 0.18


@dataclass
class Course:
    seed: int
    centre: np.ndarray          # (n, 2) points STEP apart
    heading: np.ndarray         # (n,) tangent angle
    curvature: np.ndarray       # (n,) signed, 1/m
    cones: np.ndarray           # (m, 2)
    cone_side: np.ndarray       # (m,) +1 left, -1 right, 0 start/finish marker
    lot: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax of the asphalt
    start_s: float              # arc length of the start line
    finish_s: float             # arc length of the finish line
    mirrored: bool = False
    half_width: float = HALF_WIDTH
    spacing: float | None = None

    @property
    def length(self) -> float:
        return self.finish_s - self.start_s

    def to_json(self) -> dict:
        return {"seed": self.seed, "mirrored": self.mirrored,
                "centre": np.round(self.centre[::4], 2).tolist(),
                "cones": np.round(self.cones, 2).tolist(), "lot": self.lot,
                "start": self.pose_at(self.start_s), "finish": self.pose_at(self.finish_s),
                "half_width": self.half_width, "spacing": self.spacing}

    def pose_at(self, s: float) -> list[float]:
        i = int(np.clip(round(s / STEP), 0, len(self.centre) - 1))
        return [float(self.centre[i, 0]), float(self.centre[i, 1]), float(self.heading[i])]


def _segments(rng: np.random.Generator) -> list[tuple[float, float]]:
    """(length, curvature) pieces: a run-up, a random middle, a run-out."""
    segs = [(35.0, 0.0)]
    total, last_sign = 0.0, 0
    while total < 420:
        kind = rng.choice(["straight", "arc", "arc", "slalom"])
        if kind == "straight":
            segs.append((float(rng.uniform(15, 45)), 0.0))
            total += segs[-1][0]
        elif kind == "arc":
            r = float(rng.uniform(12, 40))
            ang = math.radians(float(rng.uniform(35, 150)))
            sign = -last_sign if last_sign and rng.random() < 0.6 else rng.choice([-1, 1])
            segs.append((r * ang, sign / r))
            last_sign = sign
            total += r * ang
        else:                                   # an S: three short opposite arcs
            r = float(rng.uniform(14, 22))
            ang = math.radians(float(rng.uniform(35, 55)))
            sign = rng.choice([-1, 1])
            for k, f in ((1, 0.5), (-1, 1.0), (1, 0.5)):
                segs.append((r * ang * f, k * sign / r))
                total += r * ang * f
    segs.append((30.0, 0.0))
    return segs


def _trace(segs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts, hdg, curv = [np.zeros(2)], [0.0], [0.0]
    x, y, th = 0.0, 0.0, 0.0
    for length, k in segs:
        for _ in range(max(1, int(round(length / STEP)))):
            if k:
                th2 = th + k * STEP
                x += (math.sin(th2) - math.sin(th)) / k
                y -= (math.cos(th2) - math.cos(th)) / k
                th = th2
            else:
                x += STEP * math.cos(th)
                y += STEP * math.sin(th)
            pts.append(np.array([x, y])); hdg.append(th); curv.append(k)
    return np.array(pts), np.array(hdg), np.array(curv)


def _self_clear(centre: np.ndarray) -> bool:
    """No two parts of the course closer than two widths unless adjacent."""
    sub = centre[::4]                                  # 2 m apart
    d = np.linalg.norm(sub[:, None] - sub[None], axis=-1)
    idx = np.arange(len(sub))
    far = np.abs(idx[:, None] - idx[None]) * 2.0 > 40
    return bool((d[far] > 4 * HALF_WIDTH + 4).all())


def _cones_along(centre, heading, curvature, side: int, half: float = HALF_WIDTH, spacing: float | None = None) -> list[np.ndarray]:
    normal = np.stack([-np.sin(heading), np.cos(heading)], 1) * side
    line = centre + normal * half
    out, since = [line[0]], 0.0
    for i in range(1, len(line)):
        since += float(np.linalg.norm(line[i] - line[i - 1]))
        gap = spacing or (9.0 if abs(curvature[i]) < 1 / 60 else 4.5)
        if since >= gap:
            out.append(line[i]); since = 0.0
    return out


# Course 100 is laid out like DrivingBench's cone course (drivingbench.com/report):
# a left entry turn, a 47 m aisle with gentle bends, a right turn, a 27 m cross
# aisle, a right turn and an 18 m final aisle to the finish, 8 m wide, about 130 m
# timed. Their lot is a backwards U; the turn radii here are guesses (8-9 m).
DRIVINGBENCH = 100
DB_SEGMENTS = [(22.0, 0.0), (9 * math.pi / 2, 1 / 9), (14.0, 0.0), (40 * math.radians(12), 1 / 40),
               (40 * math.radians(12), -1 / 40), (17.0, 0.0), (8 * math.pi / 2, -1 / 8), (27.0, 0.0),
               (8 * math.pi / 2, -1 / 8), (18.0, 0.0), (22.0, 0.0)]


def make_course(seed: int, mirrored: bool = False, spacing: float | None = None) -> Course:
    """`spacing`: metres between cones along each line, everywhere; by default 9 m
    on straights and 4.5 m in bends. DrivingBench's lines are about 2.5 m apart."""
    rng = np.random.default_rng(seed)
    half = HALF_WIDTH
    if seed == DRIVINGBENCH:
        centre, heading, curv = _trace(DB_SEGMENTS)
        half = 4.0
    else:
        for _ in range(200):
            centre, heading, curv = _trace(_segments(rng))
            if _self_clear(centre):
                break
        else:
            raise RuntimeError(f"no clear course for seed {seed}")
    # Rotate so the course's long axis is not always east, for variety of sun angle.
    rot = float(rng.uniform(-math.pi, math.pi)) if seed != DRIVINGBENCH else 0.6
    c, s = math.cos(rot), math.sin(rot)
    centre = centre @ np.array([[c, s], [-s, c]])
    heading = heading + rot
    centre = centre - centre.mean(0)
    if mirrored:
        centre = centre * np.array([1, -1]); heading = -heading; curv = -curv

    start_s, finish_s = (30.0, (len(centre) - 1) * STEP - 18.0) if seed != DRIVINGBENCH else (22.0, (len(centre) - 1) * STEP - 22.0)
    cones, sides = [], []
    for side in (1, -1):
        pts = _cones_along(centre, heading, curv, side, half, spacing)
        cones += pts; sides += [side] * len(pts)
    for s_line in (start_s, finish_s):           # a tall pair at each end of the timed section
        i = int(s_line / STEP)
        n = np.array([-math.sin(heading[i]), math.cos(heading[i])])
        for side in (1, -1):
            cones.append(centre[i] + n * side * (half + 0.6)); sides.append(0)
    cones = np.array(cones)
    # The run-out ends in a wall of cones across the course: stop before it.
    i = len(centre) - 1
    n = np.array([-math.sin(heading[i]), math.cos(heading[i])])
    wall = [centre[i] + n * off for off in np.linspace(-half, half, 5)]
    cones = np.vstack([cones, wall]); sides += [-2] * len(wall)

    margin = 28.0
    lo, hi = centre.min(0) - margin, centre.max(0) + margin
    return Course(seed, centre, heading, curv, cones, np.array(sides),
                  (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])),
                  start_s, finish_s, mirrored, half, spacing)


@dataclass
class Car:
    x: float
    y: float
    yaw: float
    v: float = 0.0
    wheel: float = 0.0          # road-wheel angle, radians, + is left
    accel: float = 0.0          # actual longitudinal acceleration

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([[WHEELBASE + FRONT_OVERHANG, CAR_WIDTH / 2], [WHEELBASE + FRONT_OVERHANG, -CAR_WIDTH / 2],
                          [-REAR_OVERHANG, -CAR_WIDTH / 2], [-REAR_OVERHANG, CAR_WIDTH / 2]])
        return local @ np.array([[c, s], [-s, c]]) + [self.x, self.y]


@dataclass
class Sim:
    """One run on one course. `control(steer, accel)` sets the driver's inputs:
    steer in [-1, 1] as a fraction of full lock (+ is left), accel in m/s^2."""

    course: Course
    dt: float = 0.02
    max_time: float = 240.0
    strict: bool = False        # DrivingBench's rule: any cone touched, or the cone line crossed, ends the run
    t: float = 0.0
    steer_cmd: float = 0.0
    accel_cmd: float = 0.0
    car: Car = field(init=False)
    cone_pos: np.ndarray = field(init=False)     # (m, 3): x, y, yaw of each cone
    cone_down: np.ndarray = field(init=False)
    hits: int = 0
    idx: int = 0                                 # nearest centreline sample
    progress_s: float = 0.0                      # furthest arc length reached on course
    lateral: float = 0.0
    status: str = "running"                      # running, finished, off course, hit a cone, timeout, stopped
    started_at: float | None = None
    finished_at: float | None = None
    still_since: float = 0.0
    path: list = field(default_factory=list)
    distance: float = 0.0

    def __post_init__(self):
        x, y, yaw = self.course.pose_at(self.course.start_s - 22.0)
        self.car = Car(x, y, yaw)
        self.idx = int((self.course.start_s - 22.0) / STEP)
        self.progress_s = self.course.start_s
        rng = np.random.default_rng(self.course.seed + 7)
        self.cone_pos = np.column_stack([self.course.cones, rng.uniform(0, math.pi / 2, len(self.course.cones))])
        self.cone_down = np.zeros(len(self.course.cones), bool)

    def control(self, steer: float, accel: float) -> None:
        self.steer_cmd = float(np.clip(steer, -1, 1))
        self.accel_cmd = float(np.clip(accel, -MAX_BRAKE, MAX_ACCEL))

    @property
    def done(self) -> bool:
        return self.status != "running"

    @property
    def progress(self) -> float:
        return float(np.clip((self.progress_s - self.course.start_s) / self.course.length, 0, 1))

    @property
    def elapsed(self) -> float | None:
        """Time from the start line to the finish line, as an autocross times it."""
        if self.started_at is None:
            return None
        return (self.finished_at or self.t) - self.started_at

    def step(self) -> None:
        if self.done:
            return
        c, dt = self.car, self.dt
        target = self.steer_cmd * MAX_WHEEL
        rate = MAX_WHEEL / STEER_TIME * dt
        c.wheel += float(np.clip(target - c.wheel, -rate, rate))
        c.accel += (self.accel_cmd - c.accel) * min(1.0, dt / PEDAL_LAG)
        drag = 0.1 + 0.0004 * c.v * c.v if c.v > 0 else 0.0
        c.v = max(0.0, c.v + (c.accel - drag) * dt)
        k = math.tan(c.wheel) / WHEELBASE
        if c.v * c.v * abs(k) > GRIP:                  # understeer: the front runs wide
            k = math.copysign(GRIP / (c.v * c.v), k)
        c.yaw += c.v * k * dt
        c.x += c.v * math.cos(c.yaw) * dt
        c.y += c.v * math.sin(c.yaw) * dt
        self.distance += c.v * dt
        self.t += dt
        self._cones()
        self._score()

    def _cones(self) -> None:
        c = self.car
        near = np.flatnonzero(~self.cone_down & (np.hypot(self.cone_pos[:, 0] - c.x, self.cone_pos[:, 1] - c.y) < 6))
        if not len(near):
            return
        cs, sn = math.cos(c.yaw), math.sin(c.yaw)
        for j in near:
            dx, dy = self.cone_pos[j, 0] - c.x, self.cone_pos[j, 1] - c.y
            lx, ly = dx * cs + dy * sn, -dx * sn + dy * cs
            if -REAR_OVERHANG - CONE_R < lx < WHEELBASE + FRONT_OVERHANG + CONE_R and abs(ly) < CAR_WIDTH / 2 + CONE_R:
                self.cone_down[j] = True
                self.hits += 1
                if self.strict:
                    self.status = "hit a cone"
                push = 1.0 + 0.4 * c.v
                away = math.copysign(0.6, ly)
                self.cone_pos[j, 0] += push * cs - away * sn
                self.cone_pos[j, 1] += push * sn + away * cs
                self.cone_pos[j, 2] = c.yaw + math.copysign(1.2, ly)

    def _score(self) -> None:
        c, cr = self.car, self.course
        lo, hi = max(0, self.idx - 40), min(len(cr.centre), self.idx + 40)
        d = np.hypot(cr.centre[lo:hi, 0] - c.x, cr.centre[lo:hi, 1] - c.y)
        self.idx = lo + int(np.argmin(d))
        h = cr.heading[self.idx]
        self.lateral = float(-(c.x - cr.centre[self.idx, 0]) * math.sin(h) + (c.y - cr.centre[self.idx, 1]) * math.cos(h))
        s = self.idx * STEP
        if len(self.path) == 0 or self.t - self.path[-1][0] >= 0.2:
            self.path.append((round(self.t, 2), round(c.x, 2), round(c.y, 2), round(c.yaw, 3), round(c.v, 2)))
        if abs(self.lateral) > (min(OFF_COURSE, self.course.half_width) if self.strict else OFF_COURSE):
            self.status = "off course"
            return
        if s > self.progress_s and s - self.progress_s < 10:
            self.progress_s = s
        if self.started_at is None and self.progress_s >= cr.start_s + STEP:
            self.started_at = self.t
        if self.progress_s >= cr.finish_s:
            self.progress_s = cr.finish_s
            self.finished_at = self.t
            self.status = "finished"
            return
        if self.t >= self.max_time:
            self.status = "timeout"
        self.still_since = self.t if c.v > 0.3 else self.still_since
        if self.t - self.still_since > 20:
            self.status = "stopped"
