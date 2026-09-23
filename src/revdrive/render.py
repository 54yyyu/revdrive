"""The driver's view, drawn with OpenGL through moderngl, headless.

What is on screen is what a driver in a left-hand-drive Corolla would see from
the seat: dashboard at the bottom, the hood, then the lot. The lot is asphalt
with wear and stains, grass beyond, trees and light poles for depth, a sky with
a sun and clouds, and a shadow map so the cones cast shadows. The same frames
go to the model and to the web view; nothing about the course is drawn that a
driver could not see (no centreline, no overlay).

Everything is procedural: no textures or assets to ship.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np
from PIL import Image

from . import sim as S

from dataclasses import dataclass


@dataclass(frozen=True)
class Cam:
    """A camera on the car: position in the car frame (x forward from the rear axle,
    y left, z up), pitch, and horizontal field of view."""
    pos: tuple
    pitch: float
    hfov: float


# A driver's eyes, left seat.
DRIVER = Cam((1.05, 0.37, 1.22), math.radians(-2.0), math.radians(90.0))
# DrivingBench's cameras: a comma device at the top middle of the windshield, a
# narrow road camera and a wide one looking straight ahead. The fields of view are
# approximations of the comma's (the wide one is a fisheye there; rectilinear here).
NARROW = Cam((1.60, 0.0, 1.36), math.radians(-3.0), math.radians(52.0))
WIDE = Cam((1.60, 0.0, 1.36), math.radians(-3.0), math.radians(120.0))
# What the model is shown, by `--camera`: the straight-ahead view, and turned views
# taken from the same place. "windshield" is DrivingBench's mount with 90 deg views
# cut from its wide camera: 3.9 deg of error against 5.0 for their 52 deg narrow
# camera, and 4.4 from the driver's seat (dense cones, same frames). Turned 30 deg,
# a 90 deg view reaches 75 deg to the side, 15 deg past a 120 deg wide camera.
WINDSHIELD = Cam(WIDE.pos, WIDE.pitch, math.radians(90.0))
VIEWS = {"driver": DRIVER, "windshield": WINDSHIELD}
CAMS = {"driver": DRIVER, "narrow": NARROW, "wide": WIDE}
EYE, PITCH, HFOV = np.array(DRIVER.pos), DRIVER.pitch, DRIVER.hfov
SUN_EL, SUN_AZ = math.radians(38.0), math.radians(-35.0)
SHADOW_PX, SHADOW_R = 2048, 55.0      # 5 cm texels over the 110 m around the car
SUN = np.array([math.cos(SUN_EL) * math.cos(SUN_AZ), math.cos(SUN_EL) * math.sin(SUN_AZ), math.sin(SUN_EL)])


def srgb(*c):
    return np.array(c, "f4") ** 2.2


# ---------------------------------------------------------------- meshes
# Each mesh is an (n, 9) float32 array of triangle vertices: position, normal, colour.

def _tri_mesh(pos, col):
    """Flat-shaded triangles from (n, 3, 3) positions and (n, 3) colours."""
    pos = np.asarray(pos, "f4")
    n = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    out = np.zeros((len(pos), 3, 9), "f4")
    out[:, :, :3] = pos
    out[:, :, 3:6] = n[:, None]
    out[:, :, 6:] = np.asarray(col, "f4")[:, None]
    return out.reshape(-1, 9)


def frustum(r0, r1, z0, z1, col, seg=20, smooth=True, cap=False):
    """A cone or cylinder side from radius r0 at z0 to r1 at z1."""
    a = np.linspace(0, 2 * math.pi, seg + 1)
    slope = (r0 - r1) / (z1 - z0)
    verts = []
    for i in range(seg):
        a0, a1 = a[i], a[i + 1]
        p = [(r0 * math.cos(a0), r0 * math.sin(a0), z0), (r0 * math.cos(a1), r0 * math.sin(a1), z0),
             (r1 * math.cos(a1), r1 * math.sin(a1), z1), (r1 * math.cos(a0), r1 * math.sin(a0), z1)]
        nrm = [(math.cos(t), math.sin(t), slope) for t in (a0, a1, a1, a0)]
        for tri in ((0, 1, 2), (0, 2, 3)):
            for k in tri:
                nn = np.array(nrm[k]); nn /= np.linalg.norm(nn)
                verts.append([*p[k], *nn, *col])
        if cap and r1 > 0:
            for q in ((0, 0, z1), (r1 * math.cos(a0), r1 * math.sin(a0), z1), (r1 * math.cos(a1), r1 * math.sin(a1), z1)):
                verts.append([*q, 0, 0, 1, *col])
    return np.array(verts, "f4")


def box(x0, x1, y0, y1, z0, z1, col):
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], "f4")
    faces = [(4, 5, 6, 7), (0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for a, b, cc, d in faces:
        tris += [(c[a], c[b], c[cc]), (c[a], c[cc], c[d])]
    return _tri_mesh(tris, [col] * len(tris))


def sphere(r, col, n=8, squash=1.0, jitter=0.0, rng=None):
    tris = []
    th = np.linspace(0, math.pi, n + 1)
    ph = np.linspace(0, 2 * math.pi, 2 * n + 1)
    def p(i, j):
        v = np.array([math.sin(th[i]) * math.cos(ph[j]), math.sin(th[i]) * math.sin(ph[j]), math.cos(th[i]) * squash])
        if rng is not None and 0 < i < n:
            v *= 1 + jitter * math.sin(3 * ph[j] + 5 * th[i] + rng.uniform(0, 6))
        return v * r
    for i in range(n):
        for j in range(2 * n):
            a, b, c, d = p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    return _tri_mesh(tris, [col] * len(tris))


def move(mesh, dx=0.0, dy=0.0, dz=0.0, scale=1.0):
    m = mesh.copy()
    m[:, :3] = m[:, :3] * scale + [dx, dy, dz]
    return m


def cone_mesh():
    """A 28-inch traffic cone: black base, orange body, two reflective collars."""
    orange, white, black = srgb(1.0, 0.33, 0.04), srgb(0.93, 0.93, 0.9), srgb(0.07, 0.07, 0.07)
    r = lambda z: 0.15 - (0.15 - 0.03) * (z - 0.03) / 0.68
    parts = [box(-0.19, 0.19, -0.19, 0.19, 0.0, 0.035, black)]
    bands = [(0.035, 0.32, orange), (0.32, 0.44, white), (0.44, 0.50, orange), (0.50, 0.57, white), (0.57, 0.71, orange)]
    for z0, z1, col in bands:
        parts.append(frustum(r(z0), r(z1), z0, z1, col, seg=18, cap=z1 >= 0.71))
    return np.vstack(parts)


def tree_mesh(rng):
    trunk = srgb(0.30, 0.22, 0.15)
    if rng.random() < 0.5:                                         # conifer
        h = rng.uniform(7, 13)
        g = srgb(0.10 + rng.uniform(0, 0.05), 0.24 + rng.uniform(0, 0.08), 0.12)
        parts = [frustum(0.25, 0.18, 0, h * 0.3, trunk, seg=7)]
        for k in range(3):
            z0 = h * (0.2 + 0.25 * k)
            parts.append(frustum(h * (0.26 - 0.06 * k), 0.0, z0, z0 + h * 0.45, g, seg=9))
    else:                                                          # broadleaf
        h = rng.uniform(6, 11)
        g = srgb(0.16 + rng.uniform(0, 0.08), 0.32 + rng.uniform(0, 0.1), 0.10)
        parts = [frustum(0.3, 0.2, 0, h * 0.5, trunk, seg=7),
                 move(sphere(h * 0.32, g, n=6, squash=0.85, jitter=0.12, rng=rng), dz=h * 0.62)]
    return np.vstack(parts)


def pole_mesh():
    grey = srgb(0.55, 0.56, 0.58)
    return np.vstack([frustum(0.35, 0.35, 0, 0.6, srgb(0.6, 0.6, 0.58), seg=10, cap=True),
                      frustum(0.12, 0.08, 0.6, 9.0, grey, seg=8),
                      box(-0.15, 1.6, -0.08, 0.08, 8.8, 9.0, grey),
                      box(1.0, 1.8, -0.3, 0.3, 8.6, 8.85, srgb(0.3, 0.3, 0.32))])


def building_mesh(w, d, h, rng):
    wall = srgb(*(np.array([0.72, 0.70, 0.66]) * rng.uniform(0.8, 1.05)))
    glass = srgb(0.20, 0.25, 0.30)
    parts = [box(-w / 2, w / 2, -d / 2, d / 2, 0, h, wall),
             box(-w / 2 - 0.2, w / 2 + 0.2, -d / 2 - 0.2, d / 2 + 0.2, h, h + 0.5, srgb(0.4, 0.4, 0.4))]
    for z in np.arange(1.2, h - 1.0, 3.2):                          # window bands
        parts.append(box(-w / 2 - 0.05, w / 2 + 0.05, -d / 2 - 0.05, d / 2 + 0.05, z, z + 1.4, glass))
    return np.vstack(parts)


def rotate_z(mesh, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], "f4")
    m = mesh.copy()
    m[:, :3] = m[:, :3] @ R.T
    m[:, 3:6] = m[:, 3:6] @ R.T
    return m


def hood_mesh():
    """Hood and cowl in the car frame, crowned and dropping toward the nose."""
    paint, cowl = srgb(0.10, 0.16, 0.30), srgb(0.05, 0.05, 0.055)
    x0, x1, nx, ny = 1.62, WHEEL_FRONT, 28, 16
    def z(x, y):
        u = (x - x0) / (x1 - x0)
        w = half(x)
        return 0.985 - 0.13 * u ** 1.4 - 0.07 * (y / w) ** 2 - 0.06 * max(0.0, u - 0.85) ** 2 * 40
    def half(x):
        u = (x - x0) / (x1 - x0)
        return 0.86 - 0.22 * u ** 3
    grid = np.zeros((nx + 1, ny + 1, 3), "f4")
    for i, x in enumerate(np.linspace(x0, x1, nx + 1)):
        for j, v in enumerate(np.linspace(-1, 1, ny + 1)):
            y = v * half(x)
            grid[i, j] = (x, y, z(x, y))
    verts = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
            col = cowl if grid[i, j, 0] < x0 + 0.14 else paint
            for tri in ((a, b, c), (a, c, d)):
                t = np.array(tri)
                n = np.cross(t[1] - t[0], t[2] - t[0]); n /= np.linalg.norm(n)
                if n[2] < 0:
                    n = -n
                for p in t:
                    verts.append([*p, *n, *col])
    hood = np.array(verts, "f4")
    # Smooth the normals across the grid so the paint reflects in a curve, not facets.
    key = np.round(hood[:, :3], 4)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    acc = np.zeros((inv.max() + 1, 3), "f4")
    np.add.at(acc, inv.ravel(), hood[:, 3:6])
    acc /= np.linalg.norm(acc, axis=1, keepdims=True)
    hood[:, 3:6] = acc[inv.ravel()]
    return hood


def dash_mesh():
    return np.vstack([box(1.05, 1.55, -0.95, 0.95, 0.55, 0.955, srgb(0.035, 0.035, 0.04)),
                      box(1.55, 1.75, -0.95, 0.95, 0.55, 0.90, srgb(0.02, 0.02, 0.022))])   # cowl under the hood


WHEEL_FRONT = S.WHEELBASE + S.FRONT_OVERHANG - 0.08


# ---------------------------------------------------------------- shaders

COMMON = """
uniform vec3 sun;
uniform vec3 eye;
vec3 sky_color(vec3 d) {
    float h = clamp(d.z, -0.2, 1.0);
    vec3 zenith = vec3(0.10, 0.25, 0.62), horizon = vec3(0.62, 0.72, 0.84);
    vec3 c = mix(horizon, zenith, pow(max(h, 0.0), 0.55));
    float sd = max(dot(d, sun), 0.0);
    c += vec3(1.0, 0.85, 0.6) * (pow(sd, 8.0) * 0.25 + pow(sd, 400.0) * 8.0);
    return c;
}
vec3 fog(vec3 c, vec3 p) {
    float d = length(p - eye);
    float f = 1.0 - exp(-d * 0.0011);
    vec3 dir = normalize(p - eye);
    return mix(c, sky_color(vec3(dir.xy, 0.02)) * 0.95, f);
}
vec3 finish(vec3 c) {                       // ACES fit, then sRGB
    c *= 1.05;
    c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    return pow(c, vec3(1.0 / 2.2));
}
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * noise(p); p *= 2.03; a *= 0.5; } return s; }
uniform sampler2DShadow shadow_map;
uniform mat4 light_vp;
float lit(vec3 p, vec3 n) {
    vec4 q = light_vp * vec4(p + n * 0.03, 1.0);
    vec3 c = q.xyz / q.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) return 1.0;
    float s = 0.0, texel = 1.0 / 2048.0;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
        s += texture(shadow_map, vec3(c.xy + vec2(i, j) * texel * 1.2, c.z - 0.0004));
    return s / 9.0;
}
"""

SKY_VS = """#version 410
in vec2 in_pos; out vec2 ndc;
void main() { ndc = in_pos; gl_Position = vec4(in_pos, 0.9999, 1.0); }
"""
SKY_FS = """#version 410
in vec2 ndc; out vec4 frag;
uniform mat4 inv_vp;
""" + COMMON + """
void main() {
    vec4 a = inv_vp * vec4(ndc, -1.0, 1.0), b = inv_vp * vec4(ndc, 1.0, 1.0);
    vec3 d = normalize(b.xyz / b.w - a.xyz / a.w);
    vec3 c = sky_color(d);
    if (d.z > 0.0) {
        vec2 uv = d.xy / (d.z + 0.08) * 1.6 + vec2(3.1, 7.7);
        float cl = smoothstep(0.52, 0.85, fbm(uv) * 0.9 + fbm(uv * 3.1) * 0.25);
        c = mix(c, vec3(0.95, 0.96, 0.98) * (0.85 + 0.2 * fbm(uv * 2.0)), cl * smoothstep(0.0, 0.12, d.z) * 0.85);
    }
    frag = vec4(finish(c), 1.0);
}
"""

GROUND_VS = """#version 410
in vec2 in_pos; out vec3 wp;
uniform mat4 vp;
void main() { wp = vec3(in_pos, 0.0); gl_Position = vp * vec4(wp, 1.0); }
"""
GROUND_FS = """#version 410
in vec3 wp; out vec4 frag;
uniform vec4 lot;          // xmin, ymin, xmax, ymax
uniform vec4 lines;        // start x, y, finish x, y
uniform vec4 line_dirs;    // start heading cos, sin, finish cos, sin
""" + COMMON + """
float checker(vec2 p, vec2 at, vec2 dir) {
    vec2 rel = p - at;
    float along = dot(rel, dir), across = dot(rel, vec2(-dir.y, dir.x));
    if (abs(along) > 0.6 || abs(across) > 3.6) return -1.0;
    return mod(floor(along / 0.6 + 1.0) + floor(across / 0.6), 2.0);
}
void main() {
    vec2 p = wp.xy;
    float fw = length(fwidth(p));
    // Signed distance outside the lot's rounded rectangle.
    vec2 c = (lot.xy + lot.zw) * 0.5, h = (lot.zw - lot.xy) * 0.5;
    vec2 q = abs(p - c) - h + 6.0;
    float out_d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 6.0;
    float grain = mix(noise(p * 9.0), 0.5, smoothstep(0.05, 0.3, fw));
    float fine = mix(noise(p * 37.0), 0.5, smoothstep(0.01, 0.08, fw));
    vec3 asphalt = vec3(0.075, 0.075, 0.08) * (0.78 + 0.35 * fbm(p * 0.35) + 0.18 * grain + 0.10 * fine);
    asphalt *= 1.0 - 0.25 * smoothstep(0.62, 0.8, fbm(p * 0.08 + 11.0));            // oil and patch stains
    asphalt = mix(asphalt, vec3(0.16, 0.16, 0.165), smoothstep(0.7, 0.9, fbm(p * 0.05 - 4.0)) * 0.6);
    float crack = smoothstep(0.02, 0.0, abs(fbm(p * 0.6 + 2.0) - 0.5)) * (1.0 - smoothstep(0.02, 0.12, fw));
    asphalt *= 1.0 - 0.35 * crack;
    vec3 grass = vec3(0.09, 0.16, 0.05) * (0.7 + 0.6 * fbm(p * 0.2) + 0.25 * grain);
    grass = mix(grass, vec3(0.20, 0.19, 0.10), smoothstep(0.6, 0.85, fbm(p * 0.03 + 5.0)) * 0.6);
    vec3 curb = vec3(0.42, 0.41, 0.39) * (0.85 + 0.2 * grain);
    vec3 albedo = out_d < 0.0 ? asphalt : (out_d < 0.35 ? curb : grass);
    float ck = checker(p, lines.xy, line_dirs.xy);
    if (ck >= 0.0) albedo = mix(vec3(0.02), vec3(0.62), ck);
    ck = checker(p, lines.zw, line_dirs.zw);
    if (ck >= 0.0) albedo = mix(vec3(0.02), vec3(0.62), ck);
    vec3 n = vec3(0, 0, 1);
    float s = lit(wp, n);
    vec3 col = albedo * (vec3(1.0, 0.95, 0.85) * 2.6 * max(dot(n, sun), 0.0) * s + vec3(0.45, 0.55, 0.75) * 0.40);
    frag = vec4(finish(fog(col, wp)), 1.0);
}
"""

OBJ_VS = """#version 410
in vec3 in_pos; in vec3 in_norm; in vec3 in_col;
in vec4 inst;                 // x, y, yaw, fallen (0 or 1); static meshes pass zeros
out vec3 wp; out vec3 wn; out vec3 col;
uniform mat4 vp;
uniform mat4 model;
vec3 place(vec3 p) {
    if (inst.w > 0.5) p = vec3(p.z - 0.33, p.y, p.x + 0.15);      // lying on its side
    float c = cos(inst.z), s = sin(inst.z);
    return vec3(c * p.x - s * p.y + inst.x, s * p.x + c * p.y + inst.y, p.z);
}
vec3 turn(vec3 n) {
    if (inst.w > 0.5) n = vec3(n.z, n.y, n.x);
    float c = cos(inst.z), s = sin(inst.z);
    return vec3(c * n.x - s * n.y, s * n.x + c * n.y, n.z);
}
void main() {
    vec4 w = model * vec4(place(in_pos), 1.0);
    wp = w.xyz; wn = normalize(mat3(model) * turn(in_norm)); col = in_col;
    gl_Position = vp * w;
}
"""
OBJ_FS = """#version 410
in vec3 wp; in vec3 wn; in vec3 col; out vec4 frag;
uniform float gloss;
uniform float shadows;
""" + COMMON + """
void main() {
    vec3 n = normalize(wn);
    vec3 v = normalize(eye - wp);
    if (dot(n, v) < 0.0) n = -n;
    float s = shadows > 0.5 ? lit(wp, n) : 1.0;
    float diff = max(dot(n, sun), 0.0) * s;
    vec3 amb = mix(vec3(0.12, 0.12, 0.1), vec3(0.45, 0.55, 0.75), n.z * 0.5 + 0.5) * 0.6;
    vec3 c = col * (vec3(1.0, 0.95, 0.85) * 2.6 * diff + amb);
    vec3 r = reflect(-v, n);
    float spec = pow(max(dot(r, sun), 0.0), mix(12.0, 180.0, gloss)) * s;
    c += vec3(1.0, 0.95, 0.85) * spec * mix(0.15, 3.0, gloss);
    if (gloss > 0.0) {
        float fr = 0.04 + 0.96 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
        vec3 env = r.z > 0.0 ? sky_color(r) : vec3(0.08, 0.08, 0.08);
        c = mix(c, env * 0.55, clamp(fr * gloss + 0.10 * gloss, 0.0, 0.85));
    }
    frag = vec4(finish(fog(c, wp)), 1.0);
}
"""

DEPTH_VS = """#version 410
in vec3 in_pos; in vec4 inst;
uniform mat4 light_vp; uniform mat4 model;
vec3 place(vec3 p) {
    if (inst.w > 0.5) p = vec3(p.z - 0.33, p.y, p.x + 0.15);
    float c = cos(inst.z), s = sin(inst.z);
    return vec3(c * p.x - s * p.y + inst.x, s * p.x + c * p.y + inst.y, p.z);
}
void main() { gl_Position = light_vp * model * vec4(place(in_pos), 1.0); }
"""
DEPTH_FS = """#version 410
void main() {}
"""


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, "f8") - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(fovy / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[:3, 3] = [-(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)]
    return m


def car_matrix(car) -> np.ndarray:
    c, s = math.cos(car.yaw), math.sin(car.yaw)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:3, 3] = [car.x, car.y, 0.0]
    return m


def gl(m: np.ndarray) -> bytes:
    return np.asarray(m, "f4").T.tobytes()                      # column-major


def camera(car, width: int, height: int, cam: Cam = DRIVER) -> tuple[np.ndarray, np.ndarray]:
    """A camera's view-projection matrix and position in the world."""
    eye = (car_matrix(car) @ np.array([*cam.pos, 1.0]))[:3]
    fwd = np.array([math.cos(car.yaw) * math.cos(cam.pitch), math.sin(car.yaw) * math.cos(cam.pitch), math.sin(cam.pitch)])
    fovy = 2 * math.atan(math.tan(cam.hfov / 2) * height / width)
    return perspective(fovy, width / height, 0.05, 4000.0) @ look_at(eye, eye + fwd), eye


def project(car, points: np.ndarray, width: int, height: int, cam: Cam = DRIVER) -> np.ndarray:
    """Pixel coordinates (x right, y down) of world points (n, 3); NaN behind the camera."""
    vp, _ = camera(car, width, height, cam)
    q = np.column_stack([points, np.ones(len(points))]) @ vp.T
    out = np.full((len(points), 2), np.nan)
    ok = q[:, 3] > 1e-6
    ndc = q[ok, :2] / q[ok, 3:4]
    out[ok] = np.column_stack([(ndc[:, 0] + 1) / 2 * width, (1 - ndc[:, 1]) / 2 * height])
    return out


class Renderer:
    """Holds a GL context; call from one thread only."""

    def __init__(self, course: S.Course, seed: int | None = None):
        self.ctx = moderngl.create_standalone_context(require=410)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.course = course
        rng = np.random.default_rng(course.seed if seed is None else seed)
        ctx = self.ctx
        self.p_sky = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.p_ground = ctx.program(vertex_shader=GROUND_VS, fragment_shader=GROUND_FS)
        self.p_obj = ctx.program(vertex_shader=OBJ_VS, fragment_shader=OBJ_FS)
        self.p_depth = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)

        quad = np.array([-1, -1, 3, -1, -1, 3], "f4")
        self.sky = ctx.vertex_array(self.p_sky, [(ctx.buffer(quad), "2f", "in_pos")])
        R = 3000.0
        ground = np.array([-R, -R, R, -R, R, R, -R, -R, R, R, -R, R], "f4")
        self.ground_buf = ctx.buffer(ground)
        self.ground = ctx.vertex_array(self.p_ground, [(self.ground_buf, "2f", "in_pos")])

        zero = ctx.buffer(np.zeros(4, "f4"))
        def static(mesh, prog):
            vbo = ctx.buffer(np.ascontiguousarray(mesh, "f4"))
            fmt = "3f 3f 3f" if prog is self.p_obj else "3f 24x"
            names = ["in_pos", "in_norm", "in_col"] if prog is self.p_obj else ["in_pos"]
            return ctx.vertex_array(prog, [(vbo, fmt, *names), (zero, "4f/r", "inst")])
        scenery = self._scenery(rng)
        self.scenery = static(scenery, self.p_obj)
        self.scenery_d = static(scenery, self.p_depth)
        car_body = box(-S.REAR_OVERHANG, S.WHEELBASE + S.FRONT_OVERHANG, -S.CAR_WIDTH / 2, S.CAR_WIDTH / 2, 0.15, 1.45, srgb(0.5, 0.5, 0.5))
        self.car_d = static(car_body, self.p_depth)
        self.hood = static(hood_mesh(), self.p_obj)
        self.dash = static(dash_mesh(), self.p_obj)

        cone = np.ascontiguousarray(cone_mesh(), "f4")
        self.cone_vbo = ctx.buffer(cone)
        self.inst = ctx.buffer(reserve=len(course.cones) * 16)
        self.cones = ctx.vertex_array(self.p_obj, [(self.cone_vbo, "3f 3f 3f", "in_pos", "in_norm", "in_col"),
                                                   (self.inst, "4f/i", "inst")])
        self.cones_d = ctx.vertex_array(self.p_depth, [(self.cone_vbo, "3f 24x", "in_pos"), (self.inst, "4f/i", "inst")])

        self.shadow_tex = ctx.depth_texture((SHADOW_PX, SHADOW_PX))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.fbos: dict[tuple[int, int], tuple] = {}

        st, fi = course.pose_at(course.start_s), course.pose_at(course.finish_s)
        for p in (self.p_ground, self.p_obj, self.p_sky):
            if "sun" in p:
                p["sun"].value = tuple(SUN)
        self.p_ground["lot"].value = course.lot
        self.p_ground["lines"].value = (st[0], st[1], fi[0], fi[1])
        self.p_ground["line_dirs"].value = (math.cos(st[2]), math.sin(st[2]), math.cos(fi[2]), math.sin(fi[2]))
        for p in (self.p_ground, self.p_obj):
            p["shadow_map"].value = 0

    def _scenery(self, rng) -> np.ndarray:
        c = self.course
        lo, hi = np.array(c.lot[:2]), np.array(c.lot[2:])
        parts = []
        # A tree line around the lot, thicker further out.
        per = 2 * float(np.sum(hi - lo))
        for _ in range(int(per / 6)):
            edge = rng.integers(4)
            u = rng.uniform(0, 1)
            gap = rng.uniform(4, 30) if rng.random() < 0.8 else rng.uniform(30, 120)
            if edge == 0:   p = (lo[0] + u * (hi[0] - lo[0]), lo[1] - gap)
            elif edge == 1: p = (lo[0] + u * (hi[0] - lo[0]), hi[1] + gap)
            elif edge == 2: p = (lo[0] - gap, lo[1] + u * (hi[1] - lo[1]))
            else:           p = (hi[0] + gap, lo[1] + u * (hi[1] - lo[1]))
            parts.append(move(rotate_z(tree_mesh(rng), rng.uniform(0, 6.28)), p[0], p[1]))
        # A few buildings on one side, far enough to be backdrop.
        side = rng.integers(4)
        for k in range(rng.integers(2, 5)):
            w, d, h = rng.uniform(25, 60), rng.uniform(15, 30), rng.uniform(8, 22)
            u = (k + 0.5) / 4
            off = rng.uniform(140, 220)
            p = [(lo[0] + u * (hi[0] - lo[0]), lo[1] - off), (lo[0] + u * (hi[0] - lo[0]), hi[1] + off),
                 (lo[0] - off, lo[1] + u * (hi[1] - lo[1])), (hi[0] + off, lo[1] + u * (hi[1] - lo[1]))][side]
            parts.append(move(rotate_z(building_mesh(w, d, h, rng), rng.uniform(-0.2, 0.2)), p[0], p[1]))
        # Light poles on a grid inside the lot, clear of the course.
        pole = pole_mesh()
        for x in np.arange(lo[0] + 12, hi[0] - 6, 38):
            for y in np.arange(lo[1] + 12, hi[1] - 6, 38):
                if np.min(np.hypot(c.centre[:, 0] - x, c.centre[:, 1] - y)) > c.half_width + 7:
                    parts.append(move(rotate_z(pole, rng.uniform(0, 6.28)), x, y))
        return np.vstack(parts)

    def _fbo(self, w, h):
        if (w, h) not in self.fbos:
            ms = self.ctx.framebuffer(self.ctx.renderbuffer((w, h), 4, samples=4),
                                      self.ctx.depth_renderbuffer((w, h), samples=4))
            out = self.ctx.framebuffer(self.ctx.renderbuffer((w, h), 4))
            self.fbos[(w, h)] = (ms, out)
        return self.fbos[(w, h)]

    def render(self, sim: S.Sim, width: int = 640, height: int = 360, cam: Cam = DRIVER) -> Image.Image:
        # Several renderers can share a thread (the replay server keeps one per course);
        # draw in this one's context, or the calls land in whichever was made last.
        with self.ctx:
            return self._render(sim, width, height, cam)

    def _render(self, sim: S.Sim, width: int, height: int, cam: Cam) -> Image.Image:
        ctx, car = self.ctx, sim.car
        inst = np.column_stack([sim.cone_pos, sim.cone_down.astype("f4")]).astype("f4")
        self.inst.write(inst.tobytes())
        cm = car_matrix(car)

        # Shadow map from the sun, over the ground around the car: what is near
        # enough to see a shadow on gets a sharp one; beyond it everything is lit.
        mid = np.array([car.x + 30 * math.cos(car.yaw), car.y + 30 * math.sin(car.yaw), 0.0])
        light_vp = ortho(-SHADOW_R, SHADOW_R, -SHADOW_R, SHADOW_R, 1, 400) @ look_at(mid + SUN * 200, mid)
        for p in (self.p_ground, self.p_obj, self.p_depth):
            p["light_vp"].write(gl(light_vp))
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.viewport = (0, 0, SHADOW_PX, SHADOW_PX)
        eye_m = np.eye(4)
        self.p_depth["model"].write(gl(eye_m))
        self.scenery_d.render()
        self.cones_d.render(instances=len(inst))
        self.p_depth["model"].write(gl(cm))
        self.car_d.render()

        ms, out = self._fbo(width, height)
        ms.use()
        ctx.viewport = (0, 0, width, height)
        ms.clear(0, 0, 0, 1, depth=1.0)
        vp, eye = camera(car, width, height, cam)

        self.shadow_tex.use(0)
        for p in (self.p_sky, self.p_ground, self.p_obj):
            if "eye" in p:
                p["eye"].value = tuple(eye)
        self.p_sky["inv_vp"].write(gl(np.linalg.inv(vp)))
        ctx.disable(moderngl.DEPTH_TEST)
        self.sky.render()
        ctx.enable(moderngl.DEPTH_TEST)
        self.p_ground["vp"].write(gl(vp))
        self.ground.render()
        self.p_obj["vp"].write(gl(vp))
        self.p_obj["model"].write(gl(eye_m))
        self.p_obj["shadows"].value = 1.0
        self.p_obj["gloss"].value = 0.0
        self.scenery.render()
        self.p_obj["gloss"].value = 0.15
        self.cones.render(instances=len(inst))
        self.p_obj["model"].write(gl(cm))
        self.p_obj["shadows"].value = 0.0          # the car's own shadow volume contains them
        self.p_obj["gloss"].value = 0.7
        self.hood.render()
        self.p_obj["gloss"].value = 0.0
        self.dash.render()

        ctx.copy_framebuffer(out, ms)
        data = out.read(components=3)
        return Image.frombytes("RGB", (width, height), data).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def release(self):
        self.ctx.release()
