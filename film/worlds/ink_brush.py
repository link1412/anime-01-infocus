"""The brush. Every mark on the scroll is a ribbon or a wash mesh drawn in painter's order
through one shader (ink_mark), so ink behaves the same everywhere: bristles run out of ink
along the stroke, washes pool at their edges, and later marks can reserve what lies behind them."""
from __future__ import annotations
import numpy as np

STRIDE = 13     # pos2 uv2 sd2 a4(load,dry|mott,edge|soft,seed) b3(kind,occ,wet)
FMT = "2f 2f 2f 4f 3f"
NAMES = ["pos", "uv0", "sd", "a", "b"]


def smooth(a, b, x):
    t = np.clip((np.asarray(x, "f4") - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def wob(rng, n, knots=5, amp=1.0):
    """Smooth 1D wander (n samples), zero-mean-ish."""
    k = rng.normal(0, 1, knots + 3)
    x = np.linspace(1, knots + 1, n)
    i = np.floor(x).astype(int)
    f = x - i
    p0, p1, p2, p3 = k[i - 1], k[i], k[i + 1], k[np.minimum(i + 2, knots + 2)]
    return amp * .5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (3 * p1 - 3 * p2 + p3 - p0) * f * f * f)


def bell(u, a=.6):
    return np.maximum(np.sin(np.pi * np.clip(u, 0, 1)), 0.0) ** a


def nail(u, attack=.08, tail=.75):
    """Press hard at the start, leave in a long taper: the calligrapher's stroke."""
    return smooth(0, attack, u) * (1 - u ** (1 / max(tail, .05))) ** .55 * (1 + .25 * np.exp(-u / .12))


def sweep(u, attack=.12, release=.25):
    return smooth(0, attack, u) * smooth(0, release, 1 - u) ** .7


def bez(p0, p1, p2, m):
    """Quadratic Bezier batch: p*: (N,2) -> (N,m,2)."""
    t = np.linspace(0, 1, m, dtype="f4")[None, :, None]
    p0, p1, p2 = (np.atleast_2d(np.asarray(x, "f4"))[:, None, :] for x in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def resample(poly, ds=None, n=None):
    poly = np.asarray(poly, "f4")
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if n is None:
        n = max(int(s[-1] / ds) + 2, 2)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], 1).astype("f4")


def smooth_poly(poly, r=2):
    p = np.asarray(poly, "f4")
    if r <= 0 or len(p) < 2 * r + 1:
        return p
    k = np.hanning(2 * r + 3)[1:-1]
    k /= k.sum()
    q = np.stack([np.convolve(np.pad(p[:, i], r, mode="edge"), k, "valid") for i in (0, 1)], 1)
    return q.astype("f4")


class Marks:
    """Ordered list of marks; flushed as one triangle strip with degenerate joins."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.count = 0

    def array(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, STRIDE), "f4")
        return np.concatenate(self.chunks, 0)

    def _push(self, v):
        self.chunks.append(v.reshape(-1, STRIDE))

    # ---- ribbons -----------------------------------------------------------------------
    def strokes(self, paths, hw, load=1.0, dry=0.0, edge=.3, occ=0.0, wet=0.0, seed=None, rng=None):
        """paths (N,M,2) or (M,2); hw half width in view units, broadcast to (N,M);
        load broadcast to (N,M); dry/edge/occ/wet/seed broadcast to (N,)."""
        p = np.asarray(paths, "f4")
        if p.ndim == 2:
            p = p[None]
        N, M = p.shape[:2]
        if N == 0:
            return
        hw = np.broadcast_to(np.asarray(hw, "f4"), (N, M))
        lo = np.broadcast_to(np.asarray(load, "f4"), (N, M))
        d = np.gradient(p, axis=1)
        ln = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-7)
        nx, ny = -d[..., 1] / ln, d[..., 0] / ln
        seg = np.hypot(*np.moveaxis(p[:, 1:] - p[:, :-1], -1, 0))
        s = np.concatenate([np.zeros((N, 1), "f4"), np.cumsum(seg, 1)], 1)
        u = s / np.maximum(s[:, -1:], 1e-6)
        if seed is None:
            seed = (rng.random(N) if rng is not None else np.random.default_rng(N).random(N))
        V = 1.3      # extra room for edge wander
        col = lambda x: np.broadcast_to(np.asarray(x, "f4"), (N,))[:, None]
        dry_, edge_, occ_, wet_, seed_ = (np.broadcast_to(col(x), (N, M)) for x in (dry, edge, occ, wet, seed))
        out = np.zeros((N, 2 * M + 2, STRIDE), "f4")
        for side, off in ((1.0, 1), (-1.0, 2)):
            i = np.arange(M) * 2 + off
            ex = hw * V * side
            out[:, i, 0] = p[..., 0] + nx * ex
            out[:, i, 1] = p[..., 1] + ny * ex
            out[:, i, 2] = u
            out[:, i, 3] = V * side
            out[:, i, 4] = s
            out[:, i, 5] = ex
            out[:, i, 6] = lo
            out[:, i, 7] = dry_
            out[:, i, 8] = edge_
            out[:, i, 9] = seed_
            out[:, i, 10] = 1.0
            out[:, i, 11] = occ_
            out[:, i, 12] = wet_
        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        self._push(out)

    # ---- wash meshes -------------------------------------------------------------------
    def grid(self, P, load, e, soft=.01, occ=1.0, pool=.5, mott=.5, wet=0.0, seed=.5):
        """Structured wash mesh: P (R,C,2), per-vertex load, e (distance from wet edge, view units),
        soft (edge softness, view units), occ (how much it reserves what is behind), wet."""
        P = np.asarray(P, "f4")
        R, C = P.shape[:2]
        bc = lambda x: np.broadcast_to(np.asarray(x, "f4"), (R, C))
        load, e, soft, occ, wet, pool, mott = map(bc, (load, e, soft, occ, wet, pool, mott))
        v = np.zeros((R, C, STRIDE), "f4")
        v[..., :2] = P
        v[..., 2] = e
        v[..., 3] = pool
        v[..., 6] = load
        v[..., 7] = mott
        v[..., 8] = soft
        v[..., 9] = seed
        v[..., 10] = 0.0
        v[..., 11] = occ
        v[..., 12] = wet
        out = np.zeros((R - 1, 2 * C + 2, STRIDE), "f4")
        out[:, 1::2][:, :C] = v[:-1]
        out[:, 2::2][:, :C] = v[1:]
        out[:, 0] = out[:, 1]
        out[:, -1] = out[:, -2]
        self._push(out)

    def blob(self, c, rx, ry, load, rng, soft=.012, occ=0.0, wet=0.0, pool=.5, mott=.5, rough=.14, tilt=0.0, n=28):
        """Irregular soft dab / rock face: rings around a centre."""
        th = np.linspace(0, 2 * np.pi, n + 1)
        r = 1 + rough * (wob(rng, n + 1, knots=7) )
        r[-1] = r[0]
        ct, st = np.cos(tilt), np.sin(tilt)
        x, y = rx * r * np.cos(th), ry * r * np.sin(th)
        ring = np.stack([c[0] + ct * x - st * y, c[1] + st * x + ct * y], 1)
        self.fan(ring, c, load, soft=soft, occ=occ, wet=wet, pool=pool, mott=mott, seed=rng.random())

    def fan(self, ring, c, load, soft=.01, occ=1.0, wet=0.0, pool=.5, mott=.5, seed=.5, rings=(1.0, .82, .5, 0.0)):
        ring = np.asarray(ring, "f4")
        c = np.asarray(c, "f4")
        rr = np.asarray(rings, "f4")[:, None, None]
        P = c + (ring - c) * rr
        dist = np.hypot(*(ring - c).T)[None]
        e = (1 - rr[..., 0]) * dist
        self.grid(P, load, e, soft=soft, occ=occ, wet=wet, pool=pool, mott=mott, seed=seed)
