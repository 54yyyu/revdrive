"""One run of one policy on one course, and the record that replays it.

A policy is asked about the world every `period` seconds, with at most one
question in flight. Its answer is either a point on the ground and a speed,
which the controller drives to (`point.Tracker`), or wheel and pedal inputs
(`human`). Timing:

- realtime: the world keeps moving while the policy thinks, holding the last
  inputs, as a car would. Headless, the world is fast-forwarded by each answer's
  measured latency (or `latency`), so a run takes no longer than its model
  calls; with the web view it runs on the wall clock.
- lockstep: the world waits for every answer.

A run returns a summary (one line in the results) and a full record: the car's
pose at 20 Hz, when each cone went down and where it ended up, and every
decision with the true bearing it is graded against. The renderer is
deterministic, so the record redraws any moment of the run.
"""

from __future__ import annotations

import base64, io, math, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from PIL import Image

from . import point as P
from . import sim as S

UI_FPS = 20
UI_SIZE = (960, 540)
TRACE_HZ = 20


def jpeg(im: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@dataclass
class Obs:
    """What a policy may know, copied at the moment of the picture. The oracle
    reads `course` and `idx`; the model policies read only `v`."""
    t: float
    x: float
    y: float
    yaw: float
    v: float
    idx: int
    course: S.Course


def observe(sim: S.Sim) -> Obs:
    c = sim.car
    return Obs(sim.t, c.x, c.y, c.yaw, c.v, sim.idx, sim.course)


class Human:
    """Arrow keys from the web view."""
    name, needs_frame = "human", False

    def __init__(self):
        self.keys: dict[str, bool] = {}

    def act(self, o: Obs, frames=None) -> dict:
        k = self.keys
        steer = (0.45 if k.get("ArrowLeft") else 0.0) - (0.45 if k.get("ArrowRight") else 0.0)
        accel = 2.5 if k.get("ArrowUp") else (-6.0 if k.get("ArrowDown") else -0.3)
        return {"steer": steer, "accel": accel}


POLICIES = "rev, blind, oracle-point, oracle-point~N (N degrees of noise per answer), human (web view only)"


def make_policy(name: str, *, upstream: str | None, camera: str, latency: float | None, period: float):
    if name.startswith("oracle-point"):
        return P.OraclePoint(noise_deg=float(name.partition("~")[2] or 0), lead=(latency or 0.0) + period / 2)
    if name in ("rev", "blind"):
        if not upstream:
            raise SystemExit(f"{name} needs --upstream URL: an sglang server serving a vision model (or REV_UPSTREAM)")
        return P.Pointer(upstream, head=camera, blind=name == "blind")
    if name == "human":
        return Human()
    raise SystemExit(f"unknown policy {name!r}; one of: {POLICIES}")


def run(course: S.Course, policy, *, mode: str = "realtime", period: float = 0.25, latency: float | None = None,
        size: tuple[int, int] = (512, 288), camera: str = "windshield", strict: bool = True,
        max_time: float = 900.0, hub=None) -> tuple[dict, dict]:
    """One run. Returns (summary, full record)."""
    from .render import VIEWS, WIDE, DRIVER, Renderer
    r = Renderer(course)
    view_cam = VIEWS[camera]
    show_cam = WIDE if camera == "windshield" else DRIVER       # what people watching see
    sim = S.Sim(course, max_time=max_time, strict=strict)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{policy.name}-{course.seed}{'m' if course.mirrored else ''}"
    trace, cone_events, decisions, charged = [], [], [], []
    pending, t_req, wall_req, obs_req, frame = None, 0.0, 0.0, None, None
    next_decision, next_ui, next_trace, hits_seen = 0.0, 0.0, 0.0, 0
    pool = ThreadPoolExecutor(1)
    wall0, paused = time.perf_counter(), 0.0     # paused: wall time spent waiting in lockstep
    from . import __version__
    meta = {"id": run_id, "version": __version__, "policy": policy.name, "seed": course.seed, "mirrored": course.mirrored,
            "camera": camera, "spacing": course.spacing, "top_speed": P.TOP, "strict": strict,
            "mode": mode, "latency": latency, "period": period, "size": list(size)}
    tracker = P.Tracker()
    if hub:
        hub.stop.clear()
        hub.live = meta
        hub.publish("run", {**meta, "course": course.to_json()})

    def row():
        c = sim.car
        return [round(sim.t, 3), round(c.x, 3), round(c.y, 3), round(c.yaw, 4), round(c.v, 3),
                round(c.wheel / S.MAX_WHEEL, 4), round(sim.steer_cmd, 4), round(sim.accel_cmd, 3),
                round(sim.progress, 5), round(sim.lateral, 3), sim.hits]

    def bookkeeping() -> list:
        nonlocal next_trace, hits_seen
        new = []
        if sim.hits != hits_seen:
            known = {e[1] for e in cone_events}
            for j in np.flatnonzero(sim.cone_down):
                if int(j) not in known:
                    x, y, yaw = sim.cone_pos[j]
                    new.append([round(sim.t, 3), int(j), round(float(x), 3), round(float(y), 3), round(float(yaw), 3)])
            cone_events.extend(new)
            hits_seen = sim.hits
        if sim.t + 1e-9 >= next_trace or new:
            trace.append(row())
            next_trace = sim.t + 1 / TRACE_HZ
        return new

    def show(thinking: bool, new_cones=()):
        im = r.render(sim, *UI_SIZE, show_cam)
        if tracker.target is not None:
            im = P.draw_target(im, sim.car, tracker.target, show_cam)
        hub.frame = jpeg(im, 80)
        hub.publish("tick", {"row": row(), "status": sim.status, "elapsed": sim.elapsed,
                             "started_at": sim.started_at, "thinking": thinking, "cones": list(new_cones)})

    def apply(answer: dict):
        if "target" in answer:
            tracker.set(answer["target"], answer["speed"])
            # Graded against the true centreline from the pose of the picture; never shown to the policy.
            answer = {**answer, "truth": round(P.true_bearing(course, obs_req, obs_req.idx, answer["d"]), 4)}
        else:
            sim.control(answer["steer"], answer["accel"])
        entry = {"t": round(t_req, 3), "applied": round(sim.t, 3), **answer}
        decisions.append(entry)
        charged.append(sim.t - t_req)
        if hub:
            extra = {"frame": base64.b64encode(jpeg(frame, 80)).decode()} if frame is not None else {}
            hub.publish("decision", {**entry, **extra})

    def step():
        if tracker.target is not None:
            tracker.control(sim)
        sim.step()
        return bookkeeping()

    pending_cones: list = []
    trace.append(row())
    while not sim.done and not (hub and hub.stop.is_set()):
        if pending is None and sim.t + 1e-9 >= next_decision:
            frames = None
            if policy.needs_frame:
                # The straight-ahead view and the turned ones, all from the same camera
                # position (on a real car, crops of the wide camera).
                frames = [r.render(SimpleNamespace(car=SimpleNamespace(x=sim.car.x, y=sim.car.y, yaw=sim.car.yaw + a),
                                                   cone_pos=sim.cone_pos, cone_down=sim.cone_down), *size, view_cam)
                          for a in policy.views]
                frame = frames[len(frames) // 2]
            t_req, wall_req, obs_req = sim.t, time.perf_counter(), observe(sim)
            pending = pool.submit(policy.act, obs_req, frames)
            next_decision = t_req + period
        if pending is not None:
            if mode == "lockstep":
                t0 = time.perf_counter()
                while hub:
                    try:
                        pending.result(timeout=0.1)
                        break
                    except FuturesTimeout:
                        show(True)                            # the world is paused; keep the page alive
                paused += time.perf_counter() - t0
                apply(pending.result()); pending = None
            elif hub is None:                                  # realtime, headless: charge the latency in sim time
                answer = pending.result()
                lag = latency if latency is not None else time.perf_counter() - wall_req
                while sim.t < t_req + lag and not sim.done:
                    step()
                if not sim.done:
                    apply(answer)
                pending = None
            elif pending.done() and sim.t >= t_req + (latency or 0.0):
                apply(pending.result()); pending = None        # realtime on the wall clock
        if sim.done:
            break
        pending_cones += step()
        if hub:
            ahead = sim.t - (time.perf_counter() - wall0 - paused)
            if ahead > 0:
                time.sleep(ahead)
            if sim.t >= next_ui:
                show(pending is not None and mode == "lockstep", pending_cones)
                pending_cones, next_ui = [], sim.t + 1 / UI_FPS
    if pending is not None:
        pending.cancel()
    pool.shutdown(wait=False)
    if hub and hub.stop.is_set() and not sim.done:
        sim.status = "stopped"
    trace.append(row())
    if hub:
        show(False, pending_cones)
    r.release()

    secs = [d["seconds"] for d in decisions if "seconds" in d]
    errs = [abs(math.degrees(d["bearing"] - d["truth"])) for d in decisions if "truth" in d]
    summary = {**meta, "status": sim.status, "progress": round(sim.progress, 4),
               "time": round(sim.elapsed, 2) if sim.status == "finished" else None,
               "sim_seconds": round(sim.t, 2), "cones_hit": sim.hits, "distance": round(sim.distance, 1),
               "max_speed": round(float(max(t[4] for t in trace)), 2), "decisions": len(decisions),
               "charged_p50": round(float(np.median(charged)), 3) if charged else None,
               "model_p50": round(float(np.median(secs)), 3) if secs else None,
               "bearing_error": round(float(np.mean(errs)), 2) if errs else None,
               "course_length": round(course.length, 1), "started_at": sim.started_at,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    return summary, {**summary, "trace": trace, "cones": cone_events, "log": decisions}


def key(rec: dict) -> tuple:
    """What makes two runs the same experiment: a rerun with an existing key is skipped."""
    return tuple(rec.get(k) for k in ("policy", "seed", "mirrored", "camera", "spacing", "top_speed", "strict",
                                      "mode", "latency", "period")) + (tuple(rec.get("size", ())),)
