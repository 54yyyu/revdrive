"""The web view: watch a run live, or replay any saved one.

A saved run is a trace, not a video: the car's pose at 20 Hz, when each cone
went down and where it ended up, and every decision. The renderer is
deterministic, so any moment - including the exact picture the model was
shown - is drawn again on request. A run is tens of kilobytes and can be
scrubbed at any speed.

    GET  /                    the page
    GET  /api/runs            saved runs, newest first, and the live one
    GET  /api/run?id=         one run: summary, course, trace, cone events, decisions
    GET  /api/frame?id=&t=&w=&h=   the driver's view at time t (id=live: the latest live frame)
    GET  /events              server-sent events for the live run
    POST /api/start           {policy, seed, mirror, mode, latency}
    POST /api/stop, /api/keys
"""

from __future__ import annotations

import bisect, gzip, json, queue, threading
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import sim as S

HERE = Path(__file__).parent


class Store:
    """Saved runs: one summary line each in the jsonl, the trace beside it."""

    def __init__(self, jsonl: Path):
        self.jsonl = jsonl
        self.dir = jsonl.parent / "runs"
        self.lock = threading.Lock()

    def save(self, summary: dict, full: dict) -> None:
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.dir / f"{summary['id']}.json.gz", "wt") as f:
                json.dump(full, f)
            with self.jsonl.open("a") as f:
                f.write(json.dumps(summary) + "\n")

    def summaries(self) -> list[dict]:
        if not self.jsonl.exists():
            return []
        out = [json.loads(l) for l in self.jsonl.read_text().splitlines() if l.strip()]
        return [r for r in reversed(out) if "id" in r and (self.dir / f"{r['id']}.json.gz").exists()]

    def load(self, run_id: str) -> dict | None:
        p = self.dir / f"{Path(run_id).name}.json.gz"
        if not p.exists():
            return None
        with gzip.open(p, "rt") as f:
            return json.load(f)


def state_at(full: dict, course: S.Course, base_cones: np.ndarray, t: float):
    """The car and cones at time t, from a trace: enough for the renderer."""
    tr = full["trace"]
    ts = [r[0] for r in tr]
    i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(tr) - 1)
    a, b = tr[i], tr[min(i + 1, len(tr) - 1)]
    u = 0.0 if b[0] == a[0] else min(max((t - a[0]) / (b[0] - a[0]), 0.0), 1.0)
    dyaw = (b[3] - a[3] + np.pi) % (2 * np.pi) - np.pi
    car = SimpleNamespace(x=a[1] + u * (b[1] - a[1]), y=a[2] + u * (b[2] - a[2]), yaw=a[3] + u * dyaw)
    pos, down = base_cones.copy(), np.zeros(len(base_cones), bool)
    for te, j, x, y, yaw in full["cones"]:
        if te <= t:
            pos[j] = (x, y, yaw); down[j] = True
    return SimpleNamespace(car=car, cone_pos=pos, cone_down=down)


class Frames:
    """Draws replay frames on its own thread, which owns its GL contexts."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.cache: dict[tuple, tuple] = {}
        threading.Thread(target=self._work, daemon=True).start()

    def draw(self, full: dict, t: float, w: int, h: int) -> bytes:
        fut: Future = Future()
        self.q.put((full, t, w, h, fut))
        return fut.result(timeout=30)

    def _work(self):
        from .render import Renderer
        import io
        while True:
            full, t, w, h, fut = self.q.get()
            try:
                key = (full["seed"], full["mirrored"], full.get("spacing"))
                if key not in self.cache:
                    if len(self.cache) >= 3:
                        old = next(iter(self.cache)); self.cache.pop(old)[0].release()
                    course = S.make_course(*key)
                    self.cache[key] = (Renderer(course), course, S.Sim(course).cone_pos.copy())
                r, course, base = self.cache[key]
                from .render import VIEWS, WIDE, DRIVER
                st = state_at(full, course, base, t)
                windshield = full.get("camera", "driver") == "windshield"
                # The replay view: the wide camera for windshield runs; the model's picture: its own camera.
                cam = (WIDE if windshield else DRIVER) if w >= 640 else VIEWS[full.get("camera", "driver")]
                im = r.render(st, w, h, cam)
                if w >= 640:                               # the replay view, not the model's picture
                    dec = [d for d in full.get("log", []) if d.get("target") and d.get("applied", d["t"]) <= t]
                    if dec:
                        from .point import draw_target
                        im = draw_target(im, st.car, dec[-1]["target"], cam)
                buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
                fut.set_result(buf.getvalue())
            except Exception as e:                                  # noqa: BLE001 - reported to the page
                fut.set_exception(e)


class Hub:
    """The live run: its latest frame, its event stream, and jobs from the page."""

    def __init__(self):
        self.cv = threading.Condition()
        self.seq = 0
        self.events: list[tuple[int, str, str]] = []
        self.frame: bytes | None = None
        self.jobs: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.human = None
        self.live: dict | None = None          # {"id", "summary-so-far"} while a run is on

    def publish(self, event: str, data) -> None:
        with self.cv:
            self.seq += 1
            self.events.append((self.seq, event, json.dumps(data)))
            del self.events[:-400]
            self.cv.notify_all()

    def since(self, seq: int, timeout: float = 15.0):
        with self.cv:
            if self.seq <= seq:
                self.cv.wait(timeout)
            return [e for e in self.events if e[0] > seq]


def serve(hub: Hub, store: Store, port: int) -> ThreadingHTTPServer:
    page = HERE / "ui.html"
    frames = Frames()
    loaded: dict[str, dict] = {}

    def run_full(run_id: str) -> dict | None:
        if run_id not in loaded:
            full = store.load(run_id)
            if full is None:
                return None
            if len(loaded) > 20:
                loaded.pop(next(iter(loaded)))
            loaded[run_id] = full
        return loaded[run_id]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json", cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._json({"runs": store.summaries(), "live": hub.live})
            if u.path == "/api/run":
                full = run_full(q.get("id", ""))
                if full is None:
                    return self._json({"error": "no such run"}, 404)
                course = S.make_course(full["seed"], full["mirrored"], full.get("spacing")).to_json()
                return self._json({**full, "course": course})
            if u.path == "/api/frame":
                if q.get("id") == "live":
                    return self._send(200, hub.frame or b"", "image/jpeg")
                full = run_full(q.get("id", ""))
                if full is None:
                    return self._json({"error": "no such run"}, 404)
                w = min(int(q.get("w", 960)), 1920); h = min(int(q.get("h", 540)), 1080)
                try:
                    body = frames.draw(full, float(q.get("t", 0)), w, h)
                except Exception as e:                              # noqa: BLE001
                    return self._json({"error": str(e)}, 500)
                return self._send(200, body, "image/jpeg", cache=True)
            if u.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                seq = hub.seq
                try:
                    while True:
                        evs = hub.since(seq)
                        if not evs:
                            self.wfile.write(b": keepalive\n\n")
                        for s_, ev, data in evs:
                            seq = s_
                            self.wfile.write(f"event: {ev}\ndata: {data}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            if self.path == "/api/start":
                hub.stop.set()
                hub.jobs.put(body)
                return self._json({"ok": True})
            if self.path == "/api/stop":
                hub.stop.set()
                return self._json({"ok": True})
            if self.path == "/api/keys":
                if hub.human is not None:
                    hub.human.keys = {k: bool(v) for k, v in body.items()}
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
