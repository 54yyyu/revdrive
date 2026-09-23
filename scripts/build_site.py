"""Build the static site from saved runs: the landing page and a replay for each run worth watching.

    python scripts/build_site.py                  # results/runs.jsonl -> site/
    python scripts/build_site.py --gallery-only   # skip re-rendering videos that exist

The published runs are the benchmark protocol - the windshield camera, a cone
every 2.5 m, a 1.5 m/s ceiling, strict, real time - the latest per policy and
course. Every rev run gets a replay, and oracle-point and blind on courses 0 and
100. A replay is the run's record (trace, cones, decisions), a video of the
camera view at 20 fps drawn from it, and the model's picture at each decision.
Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse, json, math, shutil, subprocess, sys
from pathlib import Path

import numpy as np

from revdrive import point as P, render as R, sim as S
from revdrive.web import Store, state_at

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = {"camera": "windshield", "spacing": 2.5, "top_speed": 1.5, "strict": True, "mode": "realtime",
            "mirrored": False}
LATENCY = {"oracle-point": 1.0, "rev": None, "blind": None}    # the oracle is charged rev's latency
REPLAY_ALSO = {("oracle-point", 0), ("oracle-point", S.DRIVINGBENCH), ("blind", 0), ("blind", S.DRIVINGBENCH)}
VIDEO = (960, 540)


def published(store: Store) -> list[dict]:
    """The latest run per (policy, course) under the protocol."""
    best: dict[tuple, dict] = {}
    for r in store.summaries():                      # newest first
        if r["policy"] not in LATENCY or r.get("latency") != LATENCY[r["policy"]]:
            continue
        if any(r.get(k) != v for k, v in PROTOCOL.items()) or r["status"] == "stopped":
            continue
        best.setdefault((r["policy"], r["seed"]), r)
    return sorted(best.values(), key=lambda r: (r["seed"], r["policy"]))


def course_json(seed: int) -> dict:
    c = S.make_course(seed, spacing=PROTOCOL["spacing"])
    j = c.to_json()
    j["length"] = round(c.length, 1)
    j["tightest_radius"] = round(float(1 / np.abs(c.curvature).max()), 1)
    return j


def render_video(full: dict, course: S.Course, out: Path) -> None:
    cam = R.WIDE if full["camera"] == "windshield" else R.DRIVER
    r = R.Renderer(course)
    base = S.Sim(course).cone_pos.copy()
    duration = full["trace"][-1][0]
    targets = [(d["applied"], d["target"]) for d in full["log"] if d.get("target")]
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{VIDEO[0]}x{VIDEO[1]}", "-r", "20", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                           "-crf", "32", "-tune", "animation", "-g", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    k = 0
    for i in range(int(duration * 20) + 1):
        t = i / 20
        st = state_at(full, course, base, t)
        im = r.render(st, *VIDEO, cam)
        while k + 1 < len(targets) and targets[k + 1][0] <= t:
            k += 1
        if targets and targets[k][0] <= t:
            im = P.draw_target(im, st.car, targets[k][1], cam)
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError(f"ffmpeg failed on {out}")
    r.release()


def render_thumbs(full: dict, course: S.Course, out: Path) -> None:
    """The model's straight-ahead picture at each decision, small."""
    out.mkdir(parents=True, exist_ok=True)
    r = R.Renderer(course)
    base = S.Sim(course).cone_pos.copy()
    for d in full["log"]:
        st = state_at(full, course, base, d["t"])
        r.render(st, 320, 180, R.VIEWS[full["camera"]]).save(out / f"t{d['t']:.2f}.jpg", quality=72)
    r.release()


def showcase(store: Store, runs: list[dict], out: Path) -> dict | None:
    """One real decision for the landing page's walk-through: rev on course 0 in a
    bend, read well but not perfectly (the one with the largest true bearing whose
    reading is within 4 deg). Its five views are drawn again; the numbers are the
    run's own."""
    r = next((x for x in runs if x["policy"] == "rev" and x["seed"] == 0), None)
    if r is None:
        return None
    full = store.load(r["id"])
    good = [d for d in full["log"] if abs(d["truth"]) > math.radians(10) and abs(d["bearing"] - d["truth"]) < math.radians(4)]
    if not good:
        return None
    d = max(good, key=lambda d: abs(d["truth"]))
    course = S.make_course(r["seed"], r["mirrored"], r["spacing"])
    st = state_at(full, course, S.Sim(course).cone_pos.copy(), d["t"])
    rr = R.Renderer(course)
    (out / "media" / "showcase").mkdir(parents=True, exist_ok=True)
    for k, a in enumerate(P.VIEWS):
        car = type(st.car)(x=st.car.x, y=st.car.y, yaw=st.car.yaw + a)
        view = type(st)(car=car, cone_pos=st.cone_pos, cone_down=st.cone_down)
        rr.render(view, 320, 180, R.VIEWS[full["camera"]]).save(out / "media" / "showcase" / f"view{k}.jpg", quality=80)
    rr.release()
    # the neighbourhood in the car's frame (x ahead, y left), for a small map
    c, s_ = math.cos(st.car.yaw), math.sin(st.car.yaw)
    local = lambda x, y: [round((x - st.car.x) * c + (y - st.car.y) * s_, 2), round(-(x - st.car.x) * s_ + (y - st.car.y) * c, 2)]
    near = lambda pts: [local(x, y) for x, y in pts if -6 < (x - st.car.x) * c + (y - st.car.y) * s_ < 26
                        and abs(-(x - st.car.x) * s_ + (y - st.car.y) * c) < 16]
    return {"run": r["id"], "t": d["t"], "views": d["views"], "view_deg": [round(math.degrees(a)) for a in P.VIEWS],
            "bearing": d["bearing"], "truth": d["truth"], "d": d["d"], "speed": round(d["speed"], 2),
            "speed_p": d.get("speed_p"), "seconds": d.get("seconds"), "target": local(*d["target"]),
            "centre": near(course.centre[::2].tolist()), "cones": near(course.cones.tolist())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT / "results" / "runs.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "site")
    p.add_argument("--gallery-only", action="store_true", help="keep videos and thumbnails that already exist")
    a = p.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed on PATH for the replay videos")
    store = Store(a.results)
    runs = published(store)
    if not runs:
        raise SystemExit(f"no runs under the protocol in {a.results}")
    out = a.out
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "media").mkdir(parents=True, exist_ok=True)

    seeds = sorted({r["seed"] for r in runs})
    replays = [r for r in runs if r["policy"] == "rev" or (r["policy"], r["seed"]) in REPLAY_ALSO]
    board = {"protocol": PROTOCOL, "model": "Qwen3.8-27B (AWQ INT4), served by sglang, read through rev",
             "runs": runs, "courses": {str(s): course_json(s) for s in seeds}, "paths": {},
             "replays": [r["id"] for r in replays]}
    board["showcase"] = showcase(store, runs, out)
    for r in runs:                                   # the car's line for the course maps, every 0.5 s
        full = store.load(r["id"])
        board["paths"][r["id"]] = [[round(x[1], 1), round(x[2], 1)] for x in full["trace"][::10]]
    (out / "data" / "board.json").write_text(json.dumps(board, separators=(",", ":")))
    (out / "data" / "runs.json").write_text(json.dumps({"runs": replays, "live": None}))

    for r in replays:
        full = store.load(r["id"])
        course = S.make_course(r["seed"], r["mirrored"], r["spacing"])
        (out / "data" / f"{r['id']}.json").write_text(json.dumps({**full, "course": course.to_json()}, separators=(",", ":")))
        video, thumbs = out / "media" / f"{r['id']}.mp4", out / "media" / r["id"]
        if not (a.gallery_only and video.exists()):
            render_video(full, course, video)
        if not (a.gallery_only and thumbs.exists()):
            render_thumbs(full, course, thumbs)
        poster = out / "media" / f"{r['id']}.jpg"
        if not (a.gallery_only and poster.exists()):
            rr = R.Renderer(course)
            st = state_at(full, course, S.Sim(course).cone_pos.copy(), full["trace"][-1][0] * 0.6)
            rr.render(st, 640, 360, R.WIDE if full["camera"] == "windshield" else R.DRIVER).save(poster, quality=78)
            rr.release()
        print(f"  {r['policy']:13} course {r['seed']:3}  {video.stat().st_size / 1e6:5.1f} MB", file=sys.stderr)

    viewer = (ROOT / "src" / "revdrive" / "ui.html").read_text()
    marker = "<script>\nconst $ = id =>"
    assert marker in viewer, "the viewer's script moved"
    viewer = viewer.replace(marker, '<script>window.REVDRIVE_STATIC = {data: "data/", media: "media/"};</script>\n' + marker, 1)
    viewer = viewer.replace("<title>revdrive</title>", "<title>revdrive replays</title>")
    (out / "replays.html").write_text(viewer)
    shutil.copy(ROOT / "scripts" / "site" / "index.html", out / "index.html")
    (out / ".nojekyll").write_text("")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"site: {out} ({total / 1e6:.0f} MB, {len(runs)} runs, {len(replays)} replays)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
