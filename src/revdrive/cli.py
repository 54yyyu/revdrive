"""revdrive: a decision model drives a car through a cone course from what a camera sees.

    revdrive run --policy oracle-point --courses 0-9,100        # the harness's ceiling; no model needed
    revdrive run --policy rev --courses 0-9,100 --upstream URL  # the served model
    revdrive ui --upstream URL                                  # watch, replay, or drive yourself
    revdrive summary results/runs.jsonl

URL is an sglang server serving a Qwen vision model (the tokenizer is fetched to
match; see rev's README). Runs append one line each to --out and save their
record beside it; a rerun of the same experiment is skipped.
"""

from __future__ import annotations

import argparse, json, os, socket, sys
from pathlib import Path

import numpy as np

from . import point as P
from . import sim as S
from .runner import POLICIES, Human, key, make_policy, run


def courses(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def free_port(preferred: int) -> int:
    """`preferred` if nothing is listening there, else one the OS picks."""
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def summarize(path: Path) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["status"] != "stopped" and r["policy"] != "human"]
    groups: dict[tuple, list] = {}
    for r in recs:
        setting = f"{r['camera']}, {'dense' if r.get('spacing') else 'sparse'} cones, {r['top_speed']:g} m/s, " \
                  f"{'strict' if r['strict'] else 'lenient'}, {r['mode']}" + (f" +{r['latency']:g} s" if r["latency"] else "")
        groups.setdefault((r["policy"], setting), []).append(r)
    print(f"{'policy':16} {'runs':>4} {'finished':>8} {'progress':>8} {'time s':>7} {'cones':>5} {'bearing err':>11}  setting")
    for (pol, setting), rs in sorted(groups.items()):
        fin = [r for r in rs if r["status"] == "finished"]
        errs = [r["bearing_error"] for r in rs if r.get("bearing_error") is not None]
        print(f"{pol:16} {len(rs):>4} {len(fin) / len(rs):>8.0%} {np.mean([r['progress'] for r in rs]):>8.1%} "
              f"{(f'{np.mean([r['time'] for r in fin]):.0f}' if fin else '-'):>7} "
              f"{np.mean([r['cones_hit'] for r in rs]):>5.1f} {(f'{np.mean(errs):.1f} deg' if errs else '-'):>11}  {setting}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="revdrive", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--upstream", default=os.environ.get("REV_UPSTREAM"),
                        help="sglang endpoint serving a vision model, for rev and blind (or REV_UPSTREAM)")
    common.add_argument("--camera", default="windshield", choices=("windshield", "driver"),
                        help="windshield: DrivingBench's mount, the top middle of the windshield, with 90 deg views "
                             "cut from its wide camera; driver: the left seat's eyes, 90 deg")
    common.add_argument("--cone-spacing", type=float, default=2.5, metavar="M",
                        help="metres between cones along each line; 2.5 as on DrivingBench's course map, 0 for sparse "
                             "(9 on straights, 4.5 in bends)")
    common.add_argument("--top-speed", type=float, default=1.5, metavar="M/S",
                        help="the ceiling; the model picks its speed below it (DrivingBench's cars drove 0.8-1.6 m/s)")
    common.add_argument("--lenient", action="store_true",
                        help="count cones hit instead of ending the run at the first one, and allow 4 m off the "
                             "centreline instead of the cone line")
    common.add_argument("--mode", default="realtime", choices=("realtime", "lockstep"),
                        help="realtime: the car keeps moving while the model answers; lockstep: the world waits")
    common.add_argument("--latency", type=float, metavar="S", help="charge this many seconds per answer instead of the measured time")
    common.add_argument("--period", type=float, default=0.25, metavar="S", help="seconds between pictures, at most one answer in flight")
    common.add_argument("--size", default="512x288", help="the pictures the model gets")
    common.add_argument("--max-time", type=float, default=900.0, metavar="S")
    common.add_argument("--out", type=Path, default=Path("results/runs.jsonl"),
                        help="one summary line per run; each run's record goes in runs/ beside it")

    pr = sub.add_parser("run", parents=[common], help="run courses headless")
    pr.add_argument("--policy", default="oracle-point", help=POLICIES)
    pr.add_argument("--courses", default="0", help="e.g. 0-9,100 (100 is laid out after DrivingBench's course)")
    pr.add_argument("--mirror", action="store_true", help="the mirror image of each course")
    pu = sub.add_parser("ui", parents=[common], help="the web view: live runs, replays, driving yourself")
    pu.add_argument("--port", type=int, default=8765)
    ps = sub.add_parser("summary", help="a table of saved runs")
    ps.add_argument("jsonl", type=Path, nargs="?", default=Path("results/runs.jsonl"))
    a = p.parse_args(argv)

    if a.cmd == "summary":
        summarize(a.jsonl)
        return 0
    P.TOP = a.top_speed
    size = tuple(int(v) for v in a.size.lower().split("x"))
    spacing = a.cone_spacing or None
    from .web import Store
    store = Store(a.out)
    opts = dict(mode=a.mode, period=a.period, latency=a.latency, size=size, camera=a.camera,
                strict=not a.lenient, max_time=a.max_time)

    def line(s: dict) -> str:
        return (f"{s['policy']:14} course {s['seed']}{'m' if s['mirrored'] else ''}: {s['status']}, "
                f"progress {s['progress']:.1%}" + (f", {s['time']:.0f} s" if s["time"] else "") +
                f", {s['decisions']} decisions" + (f", bearing error {s['bearing_error']:.1f} deg" if s.get("bearing_error") else "") +
                (f", model {s['model_p50']:.2f} s" if s.get("model_p50") else ""))

    if a.cmd == "run":
        if a.policy == "human":
            raise SystemExit("human drives in the web view: revdrive ui")
        policy = make_policy(a.policy, upstream=a.upstream, camera=a.camera, latency=a.latency, period=a.period)
        done = {key(r) for r in store.summaries()}
        for c in courses(a.courses):
            course = S.make_course(c, a.mirror, spacing)
            probe = {"policy": policy.name, "seed": c, "mirrored": a.mirror, "camera": a.camera, "spacing": spacing,
                     "top_speed": a.top_speed, "strict": not a.lenient, "mode": a.mode, "latency": a.latency,
                     "period": a.period, "size": list(size)}
            if key(probe) in done:
                print(f"course {c}: already in {a.out}, skipped")
                continue
            summary, full = run(course, policy, **opts)
            store.save(summary, full)
            print(line(summary), flush=True)
        return 0

    from .web import Hub, serve
    hub = Hub()
    port = free_port(a.port)
    httpd = serve(hub, store, port)
    print(f"revdrive: http://127.0.0.1:{port}  (ctrl-c to stop)", flush=True)
    policies: dict[str, object] = {}
    try:
        while True:
            job = hub.jobs.get()
            name = job.get("policy", "oracle-point")
            try:
                if name not in policies:
                    policies[name] = make_policy(name, upstream=a.upstream, camera=a.camera, latency=a.latency, period=a.period)
            except SystemExit as e:
                hub.publish("problem", {"message": str(e)})
                continue
            policy = policies[name]
            hub.human = policy if isinstance(policy, Human) else None
            course = S.make_course(int(job.get("seed") or 0), bool(job.get("mirror")), spacing)
            lat = job.get("latency")
            try:
                summary, full = run(course, policy, **{**opts, "hub": hub,
                                                       "mode": "realtime" if name == "human" else job.get("mode", a.mode),
                                                       "period": 0.05 if name == "human" else a.period,
                                                       "latency": None if lat in (None, "") else float(lat),
                                                       "strict": bool(job.get("strict", not a.lenient))})
            except Exception as e:                              # noqa: BLE001 - show it in the page, keep serving
                hub.live = None
                hub.publish("problem", {"message": f"{type(e).__name__}: {e}"})
                print(f"run failed: {e}", file=sys.stderr)
                continue
            store.save(summary, full)
            hub.live = None
            hub.publish("done", summary)
            print(line(summary), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
