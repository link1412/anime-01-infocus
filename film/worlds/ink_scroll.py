"""The handscroll itself: an original composition painted stroke by stroke, baked once.

Each layer slides at its own rate. Far ranges are one breath of pale wash; near masses are built the
way a painter builds them, big wet tone first, then buttresses that overlap and reserve each other,
then cun strokes that run with the form, then contour, then dark dots last."""
from __future__ import annotations
import hashlib, pathlib, time
import numpy as np
from ..core import GL
from .ink_brush import Marks, smooth, wob, bell, nail, bez, resample, smooth_poly
from .. import fonts as _fonts
from . import ink_tex

CACHE = pathlib.Path(__file__).resolve().parents[2] / "out" / "cache"
HERE = pathlib.Path(__file__).resolve().parent

HORIZON = -.16          # 58% down the frame
SUN = (.427, 0.0)       # 62% across, 50% down
SUN_R = .128
DMAX = 4.0              # density code range

#          parallax rate, period (view units), pixels per view unit
LAYERS = {"far":   dict(rate=.16, P=8.0, ppu=512),
          "hill":  dict(rate=.30, P=10.0, ppu=640),
          "mid":   dict(rate=.55, P=13.0, ppu=800),
          "water": dict(rate=1.0, P=16.0, ppu=800),
          "near":  dict(rate=1.0, P=16.0, ppu=1000),
          "fore":  dict(rate=1.35, P=21.6, ppu=750)}
SUN_RATE = .14
BOAT = (.640, -.455)                 # hull centre at pan 0, on the water plane
BOAT_FIGURE_OFF = (.0300, .0655)     # the small rectangle, relative to the hull centre
BOAT_SPRITE = dict(size=(.46, .34), ppu=3000)
SUN_SPRITE = dict(x0=-.62, x1=.62, y0=-1.0, y1=.42, ppu=1100)


def water_rate(y):
    """The water is a ground plane: its parallax grows towards the viewer."""
    return .30 + .95 * float(np.clip((HORIZON - y) / (HORIZON + 1.0), 0, 1))


BOAT_RATE = water_rate(BOAT[1])


# ------------------------------------------------------------------ shapes
def md(keys, rng, levels=5, rough=.15):
    """Midpoint displacement of an authored polyline. Roughness differs per stretch, so
    some ridges run clean and others break, which is what separates a line from noise."""
    p = np.asarray(keys, "f4")
    for lv in range(levels):
        a, b = p[:-1], p[1:]
        d = b - a
        L = np.hypot(d[:, 0], d[:, 1])
        n = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(L, 1e-6)[:, None]
        k = rough * (.25 + 1.6 * rng.random(len(a)) ** 2)
        mid = (a + b) / 2 + n * (rng.normal(0, 1, len(a)) * k * L)[:, None]
        q = np.zeros((len(p) + len(mid), 2), "f4")
        q[0::2] = p
        q[1::2] = mid
        p = q
    return p


def crest_of(keys, rng, levels=5, rough=.15, ds=.005, sm=1):
    return resample(smooth_poly(md(keys, rng, levels, rough), sm), ds=ds)


def sample_axis(axis, w, u, v):
    """Batch: positions at (u,v) on a ribbon described by axis (A,2) and half width w (A,)."""
    A = len(axis)
    x = np.clip(u, 0, 1) * (A - 1)
    i = np.minimum(x.astype(int), A - 2)
    f = (x - i)[..., None]
    p = axis[i] * (1 - f) + axis[i + 1] * f
    d = axis[i + 1] - axis[i]
    n = np.stack([-d[..., 1], d[..., 0]], -1) / np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-6)[..., None]
    ww = (w[i] * (1 - f[..., 0]) + w[i + 1] * f[..., 0])
    return p + n * (ww * v)[..., None]


def broken_line(m, rng, poly, hw, load, dry=.4, seg=(.08, .34), gap=(.0, .05), edge=.45, occ=0.0, wet=0.0, tone_fn=None):
    """A contour drawn the way a hand draws one: several confident pulls, with lost-and-found gaps."""
    poly = np.asarray(poly, "f4")
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])
    s = rng.uniform(0, seg[0] * .3)
    while s < arc[-1]:
        ln = rng.uniform(*seg)
        i0, i1 = np.searchsorted(arc, [s, min(s + ln, arc[-1])])
        if i1 - i0 >= 4:
            n = max(int(ln / .006), 6)
            path = resample(poly[i0:i1], n=n)
            u = np.linspace(0, 1, n)
            wd = hw * rng.uniform(.65, 1.25) * nail(u, .10, rng.uniform(.5, .95)) * (1 + .22 * wob(rng, n, 4))
            lo = load * rng.uniform(.7, 1.1)
            if tone_fn is not None:
                lo = lo * tone_fn(path)
            m.strokes(path, np.maximum(wd, hw * .12), load=lo, dry=dry * rng.uniform(.6, 1.2), edge=edge,
                      seed=rng.random(), occ=occ, wet=wet)
        s += ln + rng.uniform(*gap)


def dabs(m, rng, pts, size, load, jit=.01, ang=.35, dry=.0, wet=0.0, flat=1.0, M=5):
    """Moss dots: quick pressed marks, mostly horizontal."""
    pts = np.asarray(pts, "f4").reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return
    th = rng.normal(0, ang, n)
    ln = size * rng.uniform(.9, 2.3, n) * flat
    d = np.stack([np.cos(th), np.sin(th)], 1) * ln[:, None] / 2
    c = pts + rng.normal(0, jit, (n, 2))
    paths = bez(c - d, c + rng.normal(0, size * .25, (n, 2)), c + d, M)
    u = np.linspace(0, 1, M)[None]
    hw = (size * rng.uniform(.45, 1.0, n))[:, None] * bell(u, .55)
    m.strokes(paths, hw, load=load * rng.uniform(.7, 1.15, n)[:, None] * np.ones((n, M)), dry=dry, edge=.5, wet=wet, seed=rng.random(n))


# ------------------------------------------------------------------ mountains
def far_range(m, rng, keys, base, mist, tone=.2, ink_edge=.5, levels=5, rough=.12, soft=.004, streaks=0, wet=1.0):
    """One breath: a wash with a crisp wet top edge which pools, dissolving downward into paper."""
    crest = crest_of(keys, rng, levels, rough, ds=.012, sm=2)
    x, y = crest[:, 0], crest[:, 1]
    T = np.array([0, .012, .03, .07, .14, .24, .38, .55, .75, 1.0], "f4")[:, None]
    yy = y[None] - T * np.maximum(y[None] - base, .01)
    P = np.stack([np.broadcast_to(x, yy.shape), yy], -1)
    col = 1 + .35 * wob(rng, len(x), knots=max(int((x[-1] - x[0]) / .35), 3))
    fade = smooth(mist, mist + np.maximum(y[None] - mist, .02) * .85, yy)
    ends = (smooth(x[0], x[0] + .22, x) * smooth(x[-1], x[-1] - .22, x))[None]
    load = tone * fade * col[None] * (1.0 - .35 * T) * ends
    occ = smooth(mist - .03, mist + .05, yy) * ends
    m.grid(P, load, y[None] - yy, soft=soft, occ=occ, pool=ink_edge, mott=.38, wet=wet, seed=rng.random())
    if streaks:
        # the brush dragged down the slopes while still wet
        idx = rng.integers(0, len(x), streaks)
        drop = rng.uniform(.03, .16, streaks)
        O = crest[idx] - [0, .004]
        E = O + np.stack([rng.normal(0, .5, streaks) * drop, -drop], 1)
        paths = bez(O, (O + E) / 2 + rng.normal(0, .006, O.shape), E, 8)
        hw = rng.uniform(.002, .007, streaks)[:, None] * nail(np.linspace(0, 1, 8)[None], .1, .8)
        m.strokes(paths, hw, load=tone * rng.uniform(.35, .9, streaks)[:, None] * np.ones((streaks, 8)), dry=.75, wet=wet * .6, seed=rng.random(streaks))
    return crest


def angular_axis(rng, o, e, A, kinks=4, amp=.10):
    """A fissure is not a smooth curve: straight runs and sudden turns."""
    o, e = np.asarray(o, "f4"), np.asarray(e, "f4")
    d = e - o
    L = max(float(np.hypot(*d)), 1e-4)
    n = np.array([-d[1], d[0]], "f4") / L
    ts = np.sort(rng.uniform(.08, .95, kinks))
    keys = [o]
    for t in ts:
        keys.append(o + d * (t ** 1.25) + n * rng.normal(0, amp) * L * (.25 + t))
    keys.append(e)
    p = resample(md(np.array(keys, "f4"), rng, 2, .05), n=A)
    return smooth_poly(p, 1)


def treelets(m, rng, base, h, load):
    """Distant trees are not drawn, they are touched: upright spindle dabs of different heights."""
    base = np.asarray(base, "f4").reshape(-1, 2)
    n = len(base)
    if n == 0:
        return
    h = np.broadcast_to(np.asarray(h, "f4"), (n,))
    M = 5
    tips = base + np.stack([rng.normal(0, .10, n) * h, h], 1)
    paths = bez(base - [0, .001], (base + tips) / 2 + np.stack([rng.normal(0, .03, n) * h, np.zeros(n)], 1), tips, M)
    hw = np.maximum(h * rng.uniform(.13, .26, n), .0013)[:, None] * (np.array([[.45, 1.0, .9, .6, .18]], "f4"))
    m.strokes(paths, hw, load=(load * rng.uniform(.7, 1.15, n))[:, None] * np.ones((n, M)), dry=.08, edge=.65, seed=rng.random(n))


def massif(m, a, rng, keys, base, mist, tone=1.0, shade=-1.0, spacing=.17, body=.26, cun=1.0, ledge=1.0, fiss=1.0, axe=0.0, contour=1.0,
           dots=1.0, trees=0.0, tree_h=.03, levels=5, rough=.15, scale=1.0, plane_w=(.09, .20), spread=(.15, .62), depth=None, reach=(.55, 1.02), bleed=1.0, inner=1.0):
    """A mountain modelled the way ink models form: dark fissures running down from the crest, each one
    fading across a lit plane to the next fissure; ledges break the planes; cun hangs from the ledges."""
    crest = crest_of(keys, rng, levels, rough, ds=.006, sm=0)
    x, y = crest[:, 0], crest[:, 1]
    top = float(y.max())
    apex = crest[int(y.argmax())]
    xs_sorted = np.argsort(x)

    def cy(q):
        return np.interp(q, x[xs_sorted], y[xs_sorted])

    x0_, x1_ = float(x.min()), float(x.max())
    tap = min(.14, (x1_ - x0_) * .22)

    def endf(q):
        return smooth(x0_, x0_ + tap, q[..., 0]) * smooth(x1_, x1_ - tap, q[..., 0])

    if depth is None:
        def fade(q):
            return smooth(mist, mist + (top - mist) * .50, q[..., 1]) * endf(q)
    else:
        def fade(q):
            c = cy(q[..., 0])
            return smooth(c - depth, c - depth * .28, q[..., 1]) * endf(q)

    # 1. the big wet tone, top edge pooled, reserving what is behind
    step = max(int(.022 / .006), 1)
    cr = crest[::step]
    if not np.allclose(cr[-1], crest[-1]):
        cr = np.vstack([cr, crest[-1]])
    T = np.array([0, .008, .02, .05, .10, .18, .30, .46, .64, .82, 1.0], "f4")[:, None]
    bottom = base if depth is None else cr[:, 1][None] - depth * 1.5
    yy = cr[:, 1][None] - T * np.maximum(cr[:, 1][None] - bottom, .02)
    P = np.stack([np.broadcast_to(cr[:, 0], yy.shape), yy], -1)
    col = 1 + .45 * wob(rng, len(cr), knots=max(int((cr[-1, 0] - cr[0, 0]) / .24), 3))
    fd0 = fade(P)
    load = tone * body * fd0 * np.maximum(col, .25)[None] * (.72 + .28 * (1 - T))
    occ = smooth(mist - .10, mist + .04, yy) if depth is None else np.ones_like(yy)
    m.grid(P, load, cr[:, 1][None] - yy, soft=.0035, occ=occ * endf(P), pool=.8, mott=.42, wet=bleed * (fd0 > .02), seed=rng.random())

    # 2. fissures start from the crest at irregular intervals, flanks first, central ones last
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(crest, axis=0).T))])
    org = []
    sp = rng.uniform(0, spacing)
    while sp < arc[-1]:
        i = min(int(np.searchsorted(arc, sp)), len(y) - 1)
        if fade(crest[i]) > .03:
            org.append(crest[i])
        sp += rng.uniform(.45, 1.55) * spacing
    # spurs that start inside the mass, so that the middle of a mountain is not an unmodelled blank
    n_in = int(inner * (x1_ - x0_) * max(top - mist, .05) * 26)
    for _ in range(n_in):
        q = apex[0] + rng.normal(0, (x1_ - x0_) * .21)
        c = float(cy(q))
        lo_ = (mist if depth is None else c - depth)
        yq = lo_ + (c - lo_) * rng.uniform(.25, .92)
        if x0_ < q < x1_ and fade(np.array([q, yq], "f4")) > .05:
            org.append(np.array([q, yq], "f4"))
    org = np.array(org, "f4").reshape(-1, 2)
    ribs = []
    for o in org:
        sd = np.sign(o[0] - apex[0]) or 1.0
        lim = (o[1] - (cy(o[0]) - depth)) if depth is not None else (o[1] - mist)
        drop = max(lim, .04) * rng.uniform(*reach)
        if abs(o[1] - cy(o[0])) > .01:
            drop *= rng.uniform(.45, .8)
        outx = sd * drop * rng.uniform(*spread)
        A = 30
        ax = angular_axis(rng, o - [0, .002], o + [outx, -drop], A, kinks=int(3 + drop * 7), amp=.075)
        w0 = drop * rng.uniform(*plane_w)
        uu = np.linspace(0, 1, A)
        w = w0 * (.10 + uu ** .68) * np.maximum(1 + .30 * wob(rng, A, knots=7), .35)
        ribs.append(dict(o=o, ax=ax.astype("f4"), w=np.maximum(w, .003).astype("f4"), drop=drop, sd=sd, key=abs(o[0] - apex[0]) - o[1] * .25))
    ribs.sort(key=lambda r: -r["key"])
    V = np.array([0, .08, .2, .4, .65, 1.0], "f4")
    lit_dir = 1.0 if shade < 0 else -1.0     # the plane spreads from its fissure towards the light
    for r in ribs:
        ax, w, drop = r["ax"], r["w"], r["drop"]
        A = len(ax)
        uu = np.linspace(0, 1, A, dtype="f4")[:, None]
        # a descending axis has its normal towards +x, so v>0 is +x
        Vd = V * lit_dir
        Pg = sample_axis(ax, w, np.broadcast_to(uu, (A, len(V))), np.broadcast_to(Vd, (A, len(V))).astype("f4"))
        fd = fade(Pg)
        topf = smooth(0, .05, uu)
        strength = rng.uniform(.55, 1.45)
        ramp = .13 + 2.1 * np.exp(-V[None] / .24) * strength
        loadr = tone * body * fd * topf * ramp
        e = np.maximum(w[:, None] * (1 - V[None]), .0)
        occr = .88 * topf * smooth(1.0, .55, uu) * smooth(1.0, .35, V[None]) * (fd > .01)
        m.grid(Pg, loadr, e + .0006, soft=.012 + .02 * uu, occ=occr, pool=.25, mott=.40, wet=bleed * .8 * (fd > .02), seed=rng.random())

        # the fissure itself: a heavy pull, lost lower down, sometimes split
        ku = rng.uniform(.55, 1.0)
        n = int(A * ku) + 3
        pth = sample_axis(ax, w, np.linspace(0, ku, n, dtype="f4"), np.full(n, 0.0, "f4"))
        _fissure(m, rng, pth, hw=.0058 * scale * fiss * rng.uniform(.6, 1.35), load=tone * 1.05 * fiss, fade=fade)
        # side branches run into the plane
        nb = int(fiss * drop * 7 * rng.uniform(.5, 1.5))
        for _ in range(nb):
            u0 = rng.uniform(.1, .8)
            c0 = sample_axis(ax, w, np.array(u0, "f4"), np.array(0.0, "f4"))
            th = -np.pi / 2 + lit_dir * rng.uniform(.25, 1.0)
            ln = rng.uniform(.03, .13) * scale
            e0 = c0 + np.array([np.cos(th), np.sin(th)], "f4") * ln
            bp = angular_axis(rng, c0, e0, 10, kinks=1, amp=.14)
            _fissure(a, rng, bp, hw=.0034 * scale * rng.uniform(.7, 1.2), load=tone * .8, fade=fade, seg=(.03, .12))

        # ledges: short strokes across the plane, with cun hanging below them
        nl = int(ledge * drop * 11 * rng.uniform(.6, 1.4))
        if nl > 0:
            ul = np.sort(rng.uniform(.04, .92, nl)).astype("f4")
            for u0 in ul:
                vend = rng.uniform(.35, 1.0)
                M = 8
                tt = np.linspace(0, 1, M, dtype="f4")
                uq = np.clip(u0 + tt * rng.uniform(.005, .06) * rng.choice([1.0, .4]), 0, 1).astype("f4")
                vq = (tt * vend * lit_dir).astype("f4")
                pl = sample_axis(ax, w, uq, vq) + np.stack([np.zeros(M), .002 * wob(rng, M, 3)], 1)
                fq = fade(pl)
                if fq.mean() < .03:
                    continue
                a.strokes(pl, (.0030 * scale * rng.uniform(.6, 1.2)) * nail(tt, .1, .85), load=tone * .85 * fq * rng.uniform(.6, 1.15), dry=rng.uniform(.25, .7), edge=.7, seed=rng.random())
                # hanging cun
                nh = int(cun * vend * w[int(u0 * (A - 1))] * 150 * rng.uniform(.6, 1.4))
                if nh > 0:
                    idx = rng.integers(0, M, nh)
                    st_ = pl[idx] + rng.normal(0, .0012, (nh, 2))
                    ln = rng.uniform(.010, .040, nh) * (scale ** .5) * (1 + rng.exponential(.28, nh))
                    # hang along the local downslope of the rib
                    ad = ax[min(int(u0 * (A - 1)) + 2, A - 1)] - ax[max(int(u0 * (A - 1)) - 2, 0)]
                    ad = ad / max(np.hypot(*ad), 1e-5)
                    dirs = ad[None] + np.stack([rng.normal(0, .22, nh), rng.normal(0, .12, nh)], 1)
                    en = st_ + dirs * ln[:, None]
                    M2 = 6
                    ph = bez(st_, (st_ + en) / 2 + rng.normal(0, .0028, (nh, 2)), en, M2)
                    t2 = np.linspace(0, 1, M2)[None]
                    hw = rng.uniform(.0014, .0030, nh)[:, None] * scale * nail(t2, .08, .8)
                    lo2 = tone * rng.uniform(.3, .95, nh)[:, None] * fade(ph)
                    a.strokes(ph, hw, load=lo2, dry=rng.uniform(.25, .85, nh), edge=.5, seed=rng.random(nh))

        # along the fissure, a loose gathering of cun in the dark of the plane
        area = drop * w.mean()
        nc = int(cun * area * 400 * rng.uniform(.7, 1.3))
        if nc > 0:
            M = 9
            v0 = rng.exponential(.28, nc)
            v0 = np.minimum(v0, 1.0) * lit_dir
            u0 = rng.uniform(.02, .86, nc)
            du = rng.uniform(.03, .17, nc) * (.5 + rng.random(nc))
            tt = np.linspace(0, 1, M, dtype="f4")[None]
            uq = (u0[:, None] + du[:, None] * tt).astype("f4")
            pth = sample_axis(ax, w, uq, (v0[:, None] + rng.normal(0, .05, (nc, 1)) * tt).astype("f4"))
            hw = rng.uniform(.0014, .0028, nc)[:, None] * scale * nail(tt, .1, .85) + .0002
            lo3 = tone * rng.uniform(.28, .85, nc)[:, None] * fade(pth)
            a.strokes(pth, hw, load=lo3, dry=rng.uniform(.3, .85, nc), edge=.45, seed=rng.random(nc))
        # axe-cut: broad, angled side-brush wedges
        if axe > 0:
            nx = int(axe * area * 120 * rng.uniform(.7, 1.3))
            if nx > 0:
                M = 7
                u0 = rng.uniform(.03, .85, nx).astype("f4")
                v0 = (rng.random(nx) ** 1.3 * lit_dir).astype("f4")
                c0 = sample_axis(ax, w, u0, v0)
                th = -np.pi / 2 + lit_dir * rng.uniform(.15, .95, nx)
                ln = rng.uniform(.025, .10, nx) * scale
                d = np.stack([np.cos(th), np.sin(th)], 1) * ln[:, None]
                pth = bez(c0, c0 + d * .5 + rng.normal(0, .004, c0.shape), c0 + d, M)
                tt = np.linspace(0, 1, M)[None]
                hw = rng.uniform(.005, .016, nx)[:, None] * scale * nail(tt, .06, .7)
                lo4 = tone * rng.uniform(.25, .85, nx)[:, None] * fade(pth)
                m.strokes(pth, hw, load=lo4, dry=rng.uniform(.5, .92, nx), edge=.6, seed=rng.random(nx))

    # 3. crest contour, lost and found
    broken_line(a, rng, crest, hw=.0038 * scale * contour, load=tone * 1.05 * contour, dry=.38, seg=(.05, .30), gap=(.002, .045), tone_fn=lambda q: fade(q).mean() ** .6 if False else fade(q))

    # 4. dots last and darkest: they gather in colonies; bare stretches stay bare
    if dots > 0:
        ncol = int(dots * arc[-1] * 7)
        for _ in range(ncol):
            i = rng.integers(0, len(crest))
            g = crest[i]
            if fade(g) < .55:
                continue
            k = rng.integers(3, 16)
            off = np.stack([rng.normal(0, .028, k), -rng.exponential(.016, k) * 2.1 - .003], 1)
            q = g + off
            # never a row along the crest: that reads as stitching, not moss
            q[:, 1] = np.minimum(q[:, 1], cy(q[:, 0]) - .004 - np.random.default_rng(int(i)).exponential(.016, k))
            dabs(a, rng, q, .0036 * min(scale, 1.1) * rng.uniform(.7, 1.25), tone * 1.35, jit=.001, flat=1.6, ang=.28)
        for r in ribs:
            for _ in range(rng.integers(0, 3)):
                uh = np.array(rng.uniform(.04, .55), "f4")
                g = sample_axis(r["ax"], r["w"], uh, np.array(rng.uniform(0, .6) * lit_dir, "f4"))
                if fade(g) < .3:
                    continue
                k = rng.integers(3, 11)
                q = g + rng.normal(0, .014, (k, 2))
                q[:, 1] = np.minimum(q[:, 1], cy(q[:, 0]) - .002)
                dabs(a, rng, q, .0032 * min(scale, 1.1), tone * 1.15, jit=.001, flat=1.6, ang=.3)
    if trees > 0:
        nt = int(trees * arc[-1] * 8)
        bases = []
        for _ in range(nt):
            i = rng.integers(0, len(crest))
            if fade(crest[i]) < .6:
                continue
            k = rng.integers(2, 8)
            xo = rng.normal(0, tree_h * 1.3, k)
            bases.append(np.stack([crest[i, 0] + xo, cy(crest[i, 0] + xo) - .001 - rng.exponential(.005, k)], 1))
        if bases:
            bases = np.concatenate(bases, 0)
            treelets(a, rng, bases, tree_h * rng.uniform(.35, 1.15, len(bases)) ** 1.3, tone * 1.5)
    return crest


def _fissure(m, rng, path, hw, load, fade, seg=(.05, .26)):
    path = np.asarray(path, "f4")
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    total = arc[-1]
    s = 0.0
    while s < total - .008:
        ln = min(rng.uniform(*seg), total - s)
        i0, i1 = np.searchsorted(arc, [s, s + ln])
        if i1 - i0 >= 2:
            n = max(int(ln / .005), 5)
            p = resample(path[i0:i1 + 1], n=n)
            u = np.linspace(0, 1, n)
            prog = 1 - .72 * (s + u * ln) / max(total, 1e-4)
            wd = hw * prog * nail(u, .12, rng.uniform(.55, 1.0)) * np.maximum(1 + .35 * wob(rng, n, 3), .4)
            lo = load * fade(p) * rng.uniform(.65, 1.1)
            if lo.max() > .004:
                m.strokes(p, np.maximum(wd, .0007), load=lo, dry=rng.uniform(.18, .6), edge=.75, seed=rng.random(), wet=.2)
        s += ln * rng.uniform(.92, 1.25)


# ------------------------------------------------------------------ pines
def needle_fan(m, rng, c, r, load, n=None, up=.0, lo_ang=-.30, hi_ang=np.pi + .30, squash=.72, hw=.00175):
    n = n or int(14 + r * 280)
    c = np.asarray(c, "f4")
    th = rng.uniform(lo_ang, hi_ang, n) + up
    ln = r * rng.uniform(.74, 1.05, n)
    tip = c + np.stack([np.cos(th) * ln, np.sin(th) * ln * squash], 1)
    root = c + np.stack([np.cos(th), np.sin(th) * squash], 1) * (r * .06)
    paths = bez(root, (root + tip) / 2 + rng.normal(0, r * .018, (n, 2)), tip, 4)
    u = np.array([[0, .33, .66, 1.0]], "f4")
    w = (hw * rng.uniform(.8, 1.2, n))[:, None] * (1 - u ** 1.6 * .82)
    m.strokes(paths, w, load=load * rng.uniform(.72, 1.1, n)[:, None] * np.ones((n, 4)), dry=.05, edge=.25, seed=rng.random(n))


def foliage(pa, pb, rng, c, rx, ry, tone=1.0, fan_r=.052, density=1.0):
    """Flat-topped pads of needles: a pale wet shadow, mid-ink fans behind, a few black fans on the surface."""
    c = np.asarray(c, "f4")
    pa.blob(c - [0, ry * .25], rx * 1.0, ry * .9, tone * .30, rng, soft=.022, occ=.0, wet=1.0, pool=.1, mott=.7, rough=.25, n=22)
    k = int(density * (9 + rx * ry * 2900))
    pos = rng.uniform(-1, 1, (k * 4, 2))
    pos = pos[(pos ** 2).sum(1) < 1][:k]
    # the pad is flat on top: push fans up towards the surface
    pos[:, 1] = np.abs(pos[:, 1]) ** .6 * np.sign(pos[:, 1]) * .9 + .1
    pts = c + pos * [rx, ry]
    order = np.argsort(-pts[:, 1])
    for i in order:
        needle_fan(pb, rng, pts[i] - [0, ry * .2], fan_r * rng.uniform(.8, 1.2), tone * 1.05, up=rng.normal(0, .16), n=int(12 + fan_r * 190), hw=.0018)
    nf = max(int(k * .5), 2)
    for i in rng.choice(k, nf, replace=False) if k >= nf else range(k):
        needle_fan(pb, rng, pts[i], fan_r * rng.uniform(.72, 1.08), tone * 3.0, up=rng.normal(0, .16), n=int(10 + fan_r * 170), hw=.0021)
    # the sharp joints where fans meet the twig
    dabs(pb, rng, pts[rng.integers(0, len(pts), max(len(pts) // 2, 1))], .0024, tone * 2.6, jit=.0012, flat=1.2)


def twig_path(rng, p0, ang, length, kinks=2, droop=.0, wander=.24):
    pts = [np.asarray(p0, "f4")]
    a = ang
    seg = length / (kinks + 1)
    for i in range(kinks + 1):
        a = a + rng.normal(0, wander) - droop * np.sign(np.cos(ang))
        l = seg * rng.uniform(.7, 1.3)
        pts.append(pts[-1] + np.array([np.cos(a), np.sin(a)], "f4") * l)
    return np.array(pts, "f4")


def pine_tree(pa, pb, rng, trunk, w0, limbs, tone=1.0, fan_r=.05, roots=3):
    """A trunk with a spine like one line of cursive; scaled bark; angular limbs; flat pads of needles.
    limbs: dicts t (along trunk), ang, len, w (fraction of trunk width there), pad (rx,ry), subs [(t,ang,len,(rx,ry))]."""
    A = 46
    ax = resample(np.asarray(trunk, "f4"), n=A)
    ax = smooth_poly(ax, 4).astype("f4")
    t = np.linspace(0, 1, A)
    height = float(np.hypot(*np.diff(ax, axis=0).T).sum())
    w = (w0 * (1 - t * .70) ** 1.1 * (1 + .55 * np.exp(-t / .04)) * np.maximum(1 + .10 * wob(rng, A, 9), .5)).astype("f4")
    V = np.array([-1, -.5, 0, .5, 1], "f4")
    uu = np.broadcast_to(np.linspace(0, 1, A, dtype="f4")[:, None], (A, 5))
    P = sample_axis(ax, w, uu, np.broadcast_to(V, (A, 5)))
    load = tone * (.55 + 1.5 * smooth(-.5, 1.0, V[None])) * rng.uniform(.9, 1.1)
    pb.grid(P, load * np.ones((A, 5), "f4"), w[:, None] * (1 - np.abs(V[None])) + .0004, soft=.0012, occ=1.0, pool=.5, mott=.7)
    for v_edge, hw_, lo in ((1.0, .0050, 2.8), (-1.0, .0024, 2.0)):
        pth = sample_axis(ax, w, np.linspace(0, 1, A, dtype="f4"), np.full(A, v_edge, "f4"))
        broken_line(pb, rng, pth, hw=hw_ * (w0 / .022) ** .5, load=tone * lo, dry=.16, seg=(height * .18, height * .55), gap=(0, height * .008), edge=.7)
    # bark: small arcs that hug the trunk
    ns = int(height * 260 * (w0 / .022))
    if ns > 0:
        u0 = rng.random(ns) ** 1.2 * .94
        v0 = rng.uniform(-.85, .85, ns)
        M = 5
        tt = np.linspace(0, 1, M, dtype="f4")[None]
        ln = (np.interp(u0, t, w) * rng.uniform(.7, 1.4, ns) / height)[:, None]
        uq = (u0[:, None] + ln * (tt - .5)).astype("f4")
        vq = (v0[:, None] + rng.uniform(.25, .55, ns)[:, None] * rng.choice([-1, 1], ns)[:, None] * np.sin(np.pi * tt)).astype("f4")
        pth = sample_axis(ax, w, uq, vq)
        pb.strokes(pth, rng.uniform(.0012, .0026, ns)[:, None] * bell(tt, .6), load=tone * rng.uniform(1.0, 2.8, ns)[:, None] * np.ones((ns, M)),
                   dry=rng.uniform(.0, .4, ns), edge=.5, seed=rng.random(ns))
    # roots grip the rock
    for i in range(roots):
        side = (-1) ** i
        p0 = ax[1] + [side * w[0] * .7, .004]
        rp = twig_path(rng, p0, -np.pi / 2 + side * rng.uniform(.5, 1.25), height * rng.uniform(.06, .13), kinks=2, wander=.3)
        n = 12
        path = resample(rp, n=n)
        pb.strokes(path, w0 * .45 * (1 - np.linspace(0, 1, n) * .8), load=tone * 2.4, dry=.15, edge=.6, seed=rng.random(), occ=.9)
    pads = []
    for lb in limbs:
        q = ax[int(lb["t"] * (A - 1))]
        wb = float(np.interp(lb["t"], t, w)) * lb.get("w", .62)
        poly = twig_path(rng, q, lb["ang"], lb["len"], kinks=lb.get("kinks", 2), droop=lb.get("droop", .05))
        n = 16
        path = smooth_poly(resample(poly, n=n), 1)
        u = np.linspace(0, 1, n)
        wd = wb * (1 - u * .78) ** 1.15 * np.maximum(1 + .14 * wob(rng, n, 4), .5)
        pb.strokes(path, wd, load=tone * 2.9, dry=.1, edge=.6, seed=rng.random(), occ=.95)
        if lb.get("pad"):
            pads.append((path[-1] + [lb["len"] * .03, wb * .4], lb["pad"]))
        for tb, ang2, ln2, pad2 in lb.get("subs", ()):
            i = int(tb * (n - 1))
            poly2 = twig_path(rng, path[i], ang2, ln2, kinks=1, droop=.05)
            p2 = smooth_poly(resample(poly2, n=10), 1)
            pb.strokes(p2, wd[i] * .75 * (1 - np.linspace(0, 1, 10) * .75), load=tone * 2.9, dry=.1, edge=.6, seed=rng.random(), occ=.9)
            if pad2:
                pads.append((p2[-1], pad2))
    for c, (rx, ry) in pads:
        foliage(pa, pb, rng, c, rx, ry, tone=tone, fan_r=fan_r)


def auto_pine(pa, pb, rng, base, height, lean=.2, w0=.018, bend=(.12, -.14, .09), tone=1.0, n=6, reach=.42, side=1.0, flat=.32, fan_r=.05):
    base = np.asarray(base, "f4")
    tt = np.linspace(0, 1, 9)
    b0, b1, b2 = bend
    xs = lean * height * tt + height * (b0 * np.sin(np.pi * tt) + b1 * np.sin(2 * np.pi * tt) + b2 * np.sin(3 * np.pi * tt))
    trunk = np.stack([base[0] + xs, base[1] + height * tt], 1)
    limbs = []
    for i in range(n):
        tb = .34 + .62 * (i + rng.uniform(.1, .6)) / n
        s = side if i % 2 == 0 else -side
        ln = height * reach * (1 - abs(tb - .55) * 1.1) * rng.uniform(.75, 1.2) * (1.25 if s == side else .78)
        ang = (np.pi / 2 - s * (np.pi / 2 - .16)) + rng.normal(0, .1)
        rx = height * flat * rng.uniform(.32, .46)
        subs = []
        if rng.random() < .55:
            subs.append((rng.uniform(.4, .8), ang + s * rng.uniform(.3, .7), ln * .45, (rx * .7, rx * .26)))
        limbs.append(dict(t=tb, ang=ang, len=ln, pad=(rx, rx * rng.uniform(.28, .4)), subs=subs))
    limbs.append(dict(t=.985, ang=np.pi / 2 - side * .7, len=height * .10, pad=(height * flat * .46, height * flat * .13)))
    pine_tree(pa, pb, rng, trunk, w0, limbs, tone=tone, fan_r=fan_r)


def rock(m, a, rng, keys, depth=.30, tone=1.35, shade=-1.0, spacing=.09, scale=1.3, cun=.75, axe=1.0, dots=.7, ledge=1.1, **kw):
    """Near rock: same language, bolder hand, its ink dies away below the crest."""
    return massif(m, a, rng, keys, base=-2.0, mist=-2.0, depth=depth, tone=tone, shade=shade, spacing=spacing, body=.44, cun=cun, ledge=ledge, axe=axe,
                  contour=1.35, dots=dots, levels=4, rough=.16, scale=scale, plane_w=(.20, .42), spread=(.04, .5), reach=(.6, 1.0), **kw)


# ------------------------------------------------------------------ water
def ripples(m, rng, x0, x1, y0, y1, n, load=.36, seg=(.12, .55)):
    """A few brush lines. Fewer than you think."""
    xs = rng.uniform(x0, x1, n)
    ys = rng.uniform(min(y0, y1), max(y0, y1), n)
    k = np.clip((HORIZON - ys) / (HORIZON + 1), 0, 1)
    ln = rng.uniform(*seg, n) * (.35 + k * 1.4)
    M = 12
    y_w = rng.normal(0, .0020, (n, 1)) * (1 + 2 * k[:, None])
    p0 = np.stack([xs - ln / 2, ys], 1)
    p2 = np.stack([xs + ln / 2, ys + rng.normal(0, .002, n)], 1)
    p1 = (p0 + p2) / 2 + np.concatenate([np.zeros((n, 1)), y_w], 1)
    paths = bez(p0, p1, p2, M)
    paths[..., 1] += .0016 * np.sin(np.linspace(0, 6, M)[None] * rng.uniform(.7, 1.5, (n, 1)) + rng.uniform(0, 6, (n, 1)))
    u = np.linspace(0, 1, M)[None]
    hw = (rng.uniform(.0014, .0028, n) * (.7 + 1.1 * k))[:, None] * nail(u, .15, .8)
    m.strokes(paths, hw, load=(load * rng.uniform(.5, 1.1, n) * (.55 + k * .6))[:, None] * np.ones((n, M)), dry=rng.uniform(.45, .95, n), edge=.4, seed=rng.random(n))


# ------------------------------------------------------------------ layout: the scroll
def _peaks(rng, x0, x1, base, hmin, hmax, step=(.5, 1.2), jag=.45):
    """Procedural crest keys, irregular rhythm: one or two clear summits, then long descents."""
    xs = [x0]
    while xs[-1] < x1:
        xs.append(xs[-1] + rng.uniform(*step))
    xs = np.array(xs)
    env = wob(rng, len(xs), knots=max(len(xs) // 3, 3))
    env = (env - env.min()) / max(np.ptp(env), 1e-3)
    ys = base + (hmin + (hmax - hmin) * env ** 1.3) * (1 - jag + jag * rng.random(len(xs)))
    ys[0] = ys[-1] = base
    return np.stack([xs, ys], 1)


def paint_far():
    rng = np.random.default_rng(101)
    A = Marks()
    # opening: the ridge which grows from the sea's horizon. It passes just in front of the sun's lower limb.
    keys = [(-2.6, -.17), (-2.0, -.08), (-1.5, .0), (-1.05, -.02), (-.62, .10), (-.32, .03), (-.05, -.05), (.20, -.075), (.40, -.098), (.58, -.075),
            (.82, -.03), (1.03, .055), (1.22, .105), (1.42, .03), (1.62, .08), (1.86, .03), (2.3, -.10), (2.7, -.17)]
    far_range(A, rng, keys, HORIZON - .02, HORIZON - .012, tone=.27, ink_edge=.55)
    for x0, x1, h0, h1 in ((2.9, 5.2, .10, .34), (-5.2, -2.9, .08, .30)):
        far_range(A, rng, _peaks(rng, x0, x1, HORIZON - .01, h0, h1, step=(.28, .62)), HORIZON - .02, HORIZON - .012, tone=.25, ink_edge=.55)
    return A, Marks(), dict(iters=26, dt=.11)


def paint_hill():
    rng = np.random.default_rng(202)
    A = Marks()
    keys = [(-.60, -.20), (-.20, -.13), (.25, -.17), (.72, -.16), (.98, -.10), (1.16, -.03), (1.32, .05), (1.46, .19), (1.60, .285), (1.76, .22), (1.92, .24), (2.15, .10), (2.5, -.04), (2.95, -.20)]
    far_range(A, rng, keys, HORIZON - .04, HORIZON - .03, tone=.36, ink_edge=.8, streaks=46, rough=.15)
    for x0, x1, h0, h1 in ((3.2, 4.9, .10, .42), (-4.7, -3.1, .10, .36), (-2.8, -1.2, .05, .30)):
        far_range(A, rng, _peaks(rng, x0, x1, HORIZON - .03, h0, h1, step=(.22, .55)), HORIZON - .04, HORIZON - .03, tone=.34, ink_edge=.8, streaks=int((x1 - x0) * 26), rough=.16)
    return A, Marks(), dict(iters=26, dt=.11)


MID = dict(shade=-1, spacing=.15, body=.27, cun=.45, ledge=.55, fiss=.72, contour=.62, dots=.55, trees=.5, tree_h=.024, scale=.78)


def paint_mid():
    rng = np.random.default_rng(303)
    A, B = Marks(), Marks()
    keys = [(-.98, -.17), (-.76, .10), (-.56, .16), (-.46, .36), (-.31, .455), (-.18, .36), (-.08, .22), (.08, .12), (.22, -.02), (.42, -.14), (.62, -.25)]
    massif(A, B, rng, keys, -.34, -.27, tone=.58, **MID)
    keys = [(1.38, -.26), (1.56, -.14), (1.72, .02), (1.90, .08), (2.1, .28), (2.28, .40), (2.42, .32), (2.6, .36), (2.82, .14), (3.05, .02), (3.35, -.24)]
    massif(A, B, rng, keys, -.36, -.28, tone=.56, **MID)
    for k, x0, x1, hi in ((1, 4.2, 6.1, .62), (2, -6.0, -4.3, .50), (3, -3.6, -1.9, .42)):
        keys = _peaks(rng, x0, x1, -.26, .18, hi + .16, step=(.14, .30))
        massif(A, B, rng, keys, -.36, -.29, tone=.56, **MID)
    return A, B, dict(iters=20, dt=.10)


def paint_water():
    rng = np.random.default_rng(404)
    A = Marks()
    # opening: empty water around the boat, lines gathered below the far shore
    ripples(A, rng, -.15, 1.65, -.195, -.225, 8, load=.30, seg=(.2, .55))
    ripples(A, rng, .0, 1.6, -.26, -.40, 4, load=.30)
    ripples(A, rng, .9, 1.6, -.56, -.82, 3, load=.34)
    ripples(A, rng, 2.2, 8.0, -.2, -.9, 42, load=.36)
    ripples(A, rng, -8.0, -2.4, -.2, -.9, 42, load=.36)
    return A, Marks(), dict(iters=0, dt=.1)


def paint_near():
    rng = np.random.default_rng(505)
    A, B = Marks(), Marks()
    # ---- the hero, its dark side turned from the sun
    keys = [(-2.75, -.24), (-2.42, -.02), (-2.18, .08), (-2.02, .25), (-1.86, .34), (-1.72, .30), (-1.58, .40), (-1.47, .39), (-1.385, .625), (-1.30, .52), (-1.21, .55), (-1.115, .735), (-1.02, .875), (-.945, .845),
            (-.865, .625), (-.79, .585), (-.70, .595), (-.625, .385), (-.53, .325), (-.46, .19), (-.36, .10), (-.22, -.02), (-.02, -.15), (.26, -.28)]
    massif(A, B, rng, keys, -.40, -.27, tone=1.0, shade=-1, spacing=.125, body=.30, cun=1.0, ledge=1.0, axe=.35, contour=1.0, dots=1.0, trees=.55, tree_h=.026, inner=1.15)
    # the low spit below it
    keys = [(-.90, -.455), (-.76, -.375), (-.645, -.335), (-.56, -.355), (-.455, -.320), (-.36, -.375), (-.235, -.445)]
    rock(A, B, rng, keys, depth=.085, tone=.72, spacing=.05, scale=.72, cun=.55, axe=.2, dots=.9, ledge=.5, trees=.9, tree_h=.022)
    keys = [(-.10, -.455), (-.03, -.415), (.055, -.395), (.13, -.425), (.235, -.462)]
    rock(A, B, rng, keys, depth=.05, tone=.62, spacing=.05, scale=.65, cun=.5, axe=.1, dots=.8, ledge=.4)
    # further along
    keys = [(2.35, -.38), (2.5, -.27), (2.66, -.24), (2.82, -.15), (2.95, -.20), (3.14, -.27), (3.35, -.37)]
    rock(A, B, rng, keys, depth=.14, tone=1.0, spacing=.06, scale=.85, cun=.8, axe=.6, dots=1.2)
    keys = [(3.05, -.28), (3.3, .02), (3.48, .32), (3.60, .36), (3.74, .62), (3.92, .78), (4.02, .73), (4.14, .49), (4.26, .46), (4.40, .24), (4.56, .40), (4.74, .57),
            (4.86, .52), (5.0, .28), (5.2, .10), (5.42, -.10), (5.75, -.28)]
    massif(A, B, rng, keys, -.42, -.26, tone=1.05, shade=-1, spacing=.11, body=.28, cun=.8, ledge=1.1, axe=1.1, contour=1.1, dots=.8, trees=.35, tree_h=.028, spread=(.06, .34), plane_w=(.07, .16))
    keys = [(6.2, -.40), (6.38, -.31), (6.56, -.30), (6.72, -.36), (6.98, -.41)]
    rock(A, B, rng, keys, depth=.10, tone=.9, spacing=.06, scale=.8, cun=.6, axe=.3, dots=1.2, ledge=.6)
    keys = [(-8.6, -.26), (-8.2, -.05), (-7.85, .10), (-7.55, .08), (-7.32, .20), (-7.1, .13), (-6.82, .02), (-6.5, -.10), (-6.1, -.28)]
    massif(A, B, rng, keys, -.40, -.26, tone=.9, shade=1, spacing=.15, body=.26, cun=1.3, ledge=.4, fiss=.7, contour=.9, dots=1.3, trees=.9, tree_h=.026, spread=(.3, .9), plane_w=(.12, .24))
    keys = [(-6.05, -.26), (-5.78, .05), (-5.55, .22), (-5.38, .20), (-5.18, .52), (-5.0, .68), (-4.8, .88), (-4.62, .79), (-4.46, .70), (-4.3, .47), (-4.1, .36), (-3.92, .13),
            (-3.62, .03), (-3.34, -.12), (-3.05, -.28)]
    massif(A, B, rng, keys, -.42, -.26, tone=1.0, shade=1, spacing=.14, body=.26, cun=1.0, ledge=1.0, axe=.4, contour=1.0, dots=1.0, trees=.5, tree_h=.03)
    return A, B, dict(iters=16, dt=.09)


def paint_fore():
    rng = np.random.default_rng(606)
    A, B = Marks(), Marks()
    # ---- opening: a dark outcrop in the lower left; the larger pine reaches out towards the sun
    keys = [(-2.5, -.40), (-2.1, -.36), (-1.82, -.40), (-1.62, -.44), (-1.46, -.47), (-1.26, -.50), (-1.08, -.56), (-.92, -.62), (-.74, -.72), (-.6, -.88), (-.48, -1.08)]
    rock(A, B, rng, keys, depth=.42, tone=1.5, spacing=.085, scale=1.5, cun=.55, axe=.9, dots=.9, ledge=.85)
    trunk = [(-1.400, -.49), (-1.372, -.36), (-1.325, -.24), (-1.285, -.12), (-1.295, .02), (-1.345, .14), (-1.345, .27), (-1.28, .37)]
    limbs = [dict(t=.46, ang=.05, len=.62, w=.86, kinks=3, droop=-.02, pad=(.145, .044), subs=[(.36, .70, .13, (.115, .036)), (.62, -.50, .12, (.105, .032)), (.84, .62, .09, (.095, .030))]),
             dict(t=.36, ang=np.pi - .16, len=.31, w=.70, pad=(.135, .042), subs=[(.5, np.pi - .85, .12, (.095, .032))]),
             dict(t=.60, ang=.46, len=.25, w=.62, pad=(.135, .040)),
             dict(t=.71, ang=np.pi - .38, len=.27, w=.62, pad=(.155, .046), subs=[(.5, np.pi - 1.0, .10, (.105, .034))]),
             dict(t=.85, ang=.34, len=.21, w=.62, pad=(.145, .042)),
             dict(t=.97, ang=.52, len=.10, w=.7, pad=(.195, .056), subs=[(.3, np.pi - .5, .16, (.145, .044))])]
    pine_tree(A, B, rng, trunk, .0315, limbs, tone=1.0, fan_r=.052)
    trunk = [(-1.105, -.585), (-1.045, -.545), (-.985, -.53), (-.92, -.49), (-.85, -.47), (-.78, -.425), (-.71, -.40)]
    limbs = [dict(t=.42, ang=1.25, len=.12, w=.74, pad=(.092, .030)),
             dict(t=.64, ang=-.55, len=.13, w=.70, pad=(.092, .029)),
             dict(t=.76, ang=.85, len=.15, w=.70, pad=(.110, .034)),
             dict(t=.97, ang=.12, len=.10, w=.9, pad=(.125, .036), subs=[(.4, .9, .06, (.08, .026))])]
    pine_tree(A, B, rng, trunk, .0145, limbs, tone=.95, fan_r=.042, roots=2)
    # ---- further along (rate 1.35)
    keys = [(3.9, -1.02), (4.1, -.74), (4.32, -.64), (4.52, -.60), (4.76, -.67), (5.0, -.72), (5.25, -.83), (5.5, -1.05)]
    rock(A, B, rng, keys, depth=.38, tone=1.4, spacing=.08, scale=1.4, cun=.6, axe=.9)
    auto_pine(A, B, rng, (4.46, -.62), .62, lean=-.34, w0=.016, bend=(-.12, .11, .05), side=-1.0, n=5, reach=.52)
    keys = [(8.2, -1.03), (8.5, -.70), (8.82, -.56), (9.1, -.50), (9.42, -.58), (9.74, -.76), (10.0, -1.04)]
    rock(A, B, rng, keys, depth=.38, tone=1.4, spacing=.08, scale=1.4, cun=.6, axe=.9)
    auto_pine(A, B, rng, (9.02, -.53), .74, lean=.26, w0=.019, bend=(.09, -.1, .05), side=1.0, n=6, reach=.44)
    auto_pine(A, B, rng, (9.36, -.60), .40, lean=-.22, w0=.011, bend=(-.08, .07, .04), side=-1.0, n=4, reach=.46, flat=.28, tone=.88, fan_r=.042)
    keys = [(-7.8, -1.04), (-7.45, -.72), (-7.15, -.62), (-6.8, -.62), (-6.45, -.75), (-6.1, -1.03)]
    rock(A, B, rng, keys, depth=.38, tone=1.4, spacing=.08, scale=1.4, cun=.6, axe=.9)
    auto_pine(A, B, rng, (-7.05, -.62), .68, lean=.1, w0=.017, bend=(.12, -.08, .07), side=1.0, n=6, reach=.44)
    return A, B, dict(iters=8, dt=.07)


def paint_sun():
    """Vermilion: the disc is seal paste rather than ink, thick, slightly uneven, its rim a little heavy.
    Below it, a few dry vermilion lines on the water."""
    rng = np.random.default_rng(707)
    m = Marks()
    n = 96
    th = np.linspace(0, 2 * np.pi, n + 1)
    r = SUN_R * (1 + .006 * wob(rng, n + 1, knots=9))
    r[-1] = r[0]
    ring = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    m.fan(ring, (0, 0), 1.0, soft=.0028, occ=1.0, pool=.20, mott=.14, seed=.31, rings=(1.0, .92, .7, .35, 0.0))
    # second loaded pass, dragged with a dry brush, so that the fill is not a flat vector
    k = 30
    ys = rng.uniform(-SUN_R * .92, SUN_R * .92, k)
    half = np.sqrt(np.maximum(SUN_R ** 2 - ys ** 2, 1e-5)) * rng.uniform(.70, .96, k)
    p0 = np.stack([-half, ys], 1)
    p2 = np.stack([half, ys + rng.normal(0, .004, k)], 1)
    paths = bez(p0, (p0 + p2) / 2 + rng.normal(0, .002, (k, 2)), p2, 12)
    u = np.linspace(0, 1, 12)[None]
    m.strokes(paths, rng.uniform(.006, .016, k)[:, None] * bell(u, .4), load=rng.uniform(.10, .24, k)[:, None] * np.ones((k, 12)), dry=rng.uniform(.5, .9, k), edge=.4, seed=rng.random(k))
    # reflection, in sun-relative coordinates: broken, narrowing away
    k = 11
    ys = np.sort(rng.uniform(-.72, HORIZON - .055, k))[::-1]
    fall = np.clip((HORIZON - ys) / .7, 0, 1)
    ln = (.13 + .17 * rng.random(k)) * (1.1 - fall * .55)
    xs = rng.normal(0, .018, k)
    p0 = np.stack([xs - ln / 2, ys], 1)
    p2 = np.stack([xs + ln / 2, ys + rng.normal(0, .0015, k)], 1)
    paths = bez(p0, (p0 + p2) / 2 + rng.normal(0, .002, (k, 2)), p2, 12)
    m.strokes(paths, (rng.uniform(.0018, .0034, k) * (1 + fall))[:, None] * nail(u, .15, .85), load=(rng.uniform(.16, .40, k) * (1 - fall * .55))[:, None] * np.ones((k, 12)),
              dry=rng.uniform(.5, .9, k), edge=.4, seed=rng.random(k))
    return m


def paint_boat():
    """A skiff, an oar, and someone holding up a small rectangle. They are filming us."""
    rng = np.random.default_rng(808)
    m = Marks()
    # all coordinates relative to the hull centre
    L = .092
    # reflection under the hull, then ripples, drawn first
    for k, (x0, x1, y) in enumerate(((-.15, -.07, -.019), (.075, .165, -.022), (-.06, .07, -.024), (-.03, .05, -.035), (-.12, .03, -.047))):
        n = 14
        p = np.stack([np.linspace(x0, x1, n), y + .0012 * wob(rng, n, 4)], 1)
        m.strokes(p, .0012 * nail(np.linspace(0, 1, n), .12, .8), load=.55 - k * .05, dry=.55, edge=.3, seed=rng.random())
    # oar
    n = 12
    oar = bez([(.075, .010)], [(.115, -.002)], [(.142, -.018)], n)
    m.strokes(oar, .00135 * (1 - .35 * np.linspace(0, 1, n)), load=1.9, dry=.05, edge=.3, seed=.2)
    # hull: a single loaded sweep. Deck first, reserving; then the belly
    n = 36
    x = np.linspace(-L, L, n)
    ux = x / L
    top = .0048 + .0115 * np.abs(ux) ** 2.6
    bot = -.0125 * (1 - np.abs(ux) ** 2.1) + .0115 * np.abs(ux) ** 2.6
    P = np.stack([np.stack([x, top], 1), np.stack([x, (top + bot) / 2], 1), np.stack([x, bot], 1)], 0)
    m.grid(P, np.array([.6, 1.5, 2.8], "f4")[:, None] * np.ones((3, n), "f4"), .002, soft=.0009, occ=1.0, pool=.4, mott=.25)
    m.strokes(np.stack([x, bot], 1), .0021 * bell((ux + 1) / 2, .4), load=3.0, dry=.02, edge=.25, seed=.1, occ=1.0)
    m.strokes(np.stack([x, top], 1), .0012 * bell((ux + 1) / 2, .4), load=2.2, dry=.05, edge=.25, seed=.4)
    # ---- the figure, standing forward of centre, facing us
    fx = .012
    fy = .0050
    H = .056
    # robe: two pale-side strokes then a dark one
    robe_l = bez([(fx - .0030, fy + H * .60)], [(fx - .0056, fy + H * .3)], [(fx - .0072, fy)], 14)
    robe_r = bez([(fx + .0034, fy + H * .60)], [(fx + .0048, fy + H * .3)], [(fx + .0060, fy)], 14)
    body = np.concatenate([robe_l, robe_r[:, ::-1]], 1)[0]
    m.fan(body, (fx, fy + H * .3), np.array(.55, "f4"), soft=.0005, occ=1.0, pool=.5, mott=.2, seed=.6, rings=(1.0, .5, 0.0))
    m.strokes(robe_l, .00105 * nail(np.linspace(0, 1, 14), .1, .9), load=2.6, dry=.05, seed=.3)
    m.strokes(robe_r, .00090 * nail(np.linspace(0, 1, 14), .1, .9), load=2.2, dry=.05, seed=.7)
    # head and wide bamboo hat
    hx, hy = fx, fy + H * .66
    th = np.linspace(0, 2 * np.pi, 19)
    m.fan(np.stack([hx + .0030 * np.cos(th), hy + .0034 * np.sin(th)], 1), (hx, hy), np.array(.25, "f4"), soft=.0004, occ=1.0, pool=.2, mott=.1, seed=.8, rings=(1.0, 0.0))
    hat = np.array([[hx - .0118, hy + .0020], [hx, hy + .0118], [hx + .0118, hy + .0020], [hx, hy + .0002], [hx - .0118, hy + .0020]], "f4")
    m.fan(resample(hat, n=24), (hx, hy + .0045), np.array(2.4, "f4"), soft=.0004, occ=1.0, pool=.3, mott=.2, seed=.15, rings=(1.0, 0.0))
    # the far arm hangs; the near arm rises and holds the thing up beside the face
    m.strokes(bez([(fx - .0030, fy + H * .56)], [(fx - .0060, fy + H * .44)], [(fx - .0052, fy + H * .30)], 10), .00085, load=2.4, dry=.03, seed=.21)
    rx, ry = BOAT_FIGURE_OFF
    m.strokes(bez([(fx + .0030, fy + H * .56)], [(fx + .0085, fy + H * .47)], [(rx - .0020, ry - .0090)], 10), .00090, load=2.6, dry=.03, seed=.22)
    # the rectangle: reserved paper, a thin hard outline and one small dark dot, the lens
    w, h = .0074, .0106
    rect = np.array([[rx - w, ry - h], [rx + w, ry - h], [rx + w, ry + h], [rx - w, ry + h], [rx - w, ry - h]], "f4") + [[.0, .0]]
    m.fan(resample(rect, n=32), (rx, ry), np.array(.015, "f4"), soft=.00025, occ=1.0, pool=.0, mott=.0, seed=.9, rings=(1.0, .5, 0.0))
    m.strokes(resample(rect, n=48), .00078, load=3.2, dry=.0, edge=.1, seed=.5, occ=.6)
    th = np.linspace(0, 2 * np.pi, 13)
    m.fan(np.stack([rx + w * .50 + .0013 * np.cos(th), ry + h * .58 + .0013 * np.sin(th)], 1), (rx + w * .50, ry + h * .58), np.array(3.2, "f4"), soft=.0003, occ=1.0, pool=.0, mott=.0, seed=.3, rings=(1.0, 0.0))
    return m


PAINTERS = dict(far=paint_far, hill=paint_hill, mid=paint_mid, water=paint_water, near=paint_near, fore=paint_fore)


# ------------------------------------------------------------------ bake
def source_key():
    h = hashlib.sha1()
    for f in ("ink_brush.py", "ink_scroll.py", "ink_tex.py"):
        h.update((HERE / f).read_bytes())
    for f in ("ink_mark.vert", "ink_mark.frag", "ink_bleed.frag", "ink_kprep.frag"):
        h.update((HERE.parent / "shaders" / f).read_bytes())
    return h.hexdigest()[:10]


# ---- the painter's inscription: an original couplet, brushed on the sheet itself, not one of the marks. It reads
# "setting sun, one boat; ten thousand mountains, all characters", and it is where the painting gives itself away.
INSCRIPTION = (("落日一舟", 2.06, (1.75, 1.35, 1.10, .92)), ("万山皆字", 1.86, (1.65, 1.30, 1.05, 1.30)))     # right column first
INSCRIPTION_CH = .168
INSCRIPTION_KEY = (1.86, .22)   # a thick stroke of the last character sits exactly here
SEAL = dict(ch="字", c=(1.772, .172), size=.058)


def _ink_of_mask(a, load, seed, px):
    """Stroke mask -> ink density: rough wet edge, pooled rim, dry where the load is low, a little bleed."""
    from ..hud_ink import _filtered, _blur, sstep
    n = 1024
    h, w = a.shape
    fib = np.tile(_filtered(n, seed, 2.0, 5.0), (h // n + 1, w // n + 1))[:h, :w]
    dry = np.tile(_filtered(n, seed + 1, 9.0, 2.2), (h // n + 1, w // n + 1))[:h, :w]
    cov = sstep(.30, .56, _blur(a, 1.1 * px) + (fib - .5) * .18)
    thin = np.clip(1.5 - load, 0, 1)
    cov *= 1 - sstep(.86 - .34 * thin, .95 - .30 * thin, dry) * (.35 + .6 * thin)
    heart = _blur(a, 6.0 * px)
    return cov * load * (.80 + .55 * np.clip(1 - heart, 0, 1)) + _blur(a, 7.0 * px) * (.3 + 1.2 * fib) * .07 * load


def inscribe(raw, ppu):
    """Brush the couplet into the near layer's raw density (row 0 is y=-1). Its reserve channel stays untouched."""
    from PIL import Image, ImageDraw
    from ..hud_ink import _blur
    x0, x1, y0, y1 = 1.60, 2.32, .06, 1.0
    W, H = int((x1 - x0) * ppu), int((y1 - y0) * ppu)
    size = int(INSCRIPTION_CH * ppu * .98)
    font = _fonts.font("Xingkai.ttc", size, 0)
    rng = np.random.default_rng(1704)
    box = int(size * 1.7)

    def glyph(ch, sc, rot):
        im = Image.new("L", (box, box), 0)
        ImageDraw.Draw(im).text((box / 2, box / 2), ch, font=font, fill=255, anchor="mm")
        return im.rotate(rot, resample=Image.BICUBIC).resize((int(box * sc),) * 2, Image.BICUBIC)

    # where is the last character thick enough to carry the key?
    key = glyph("字", 1.0, 1.5)
    k = _blur(np.asarray(key, "f4") / 255, .006 * ppu)
    yy, xx = np.mgrid[0:k.shape[0], 0:k.shape[1]]
    k = k * np.exp(-(((xx - box / 2) ** 2 + (yy - box / 2) ** 2) / (.35 * size) ** 2))
    ky, kx = np.unravel_index(np.argmax(k), k.shape)
    cy_last = INSCRIPTION_KEY[1] + (ky - box / 2) / ppu
    cx_shift = -(kx - box / 2) / ppu
    canvas = np.zeros((H, W), "f4")
    for col, (text, X, loads) in enumerate(INSCRIPTION):
        for i, (ch, load) in enumerate(zip(text, loads)):
            last = (col == 1 and i == 3)
            cy = cy_last + (3 - i) * INSCRIPTION_CH * 1.0
            cx = X + (cx_shift if col == 1 else 0.0)
            sc, rot = 1.0, 1.5
            if not last:
                cx += rng.normal(0, .0045)
                cy += rng.normal(0, .003) + (.012 if col == 0 else 0.0)
                sc, rot = rng.uniform(.93, 1.05), rng.uniform(-2.5, 2.5)
            im = glyph(ch, sc, rot)
            a = np.asarray(im, "f4") / 255
            d = _ink_of_mask(a, load, 300 + col * 10 + i, ppu / 1000)
            px = int((cx - x0) * ppu - a.shape[1] / 2)
            py = int((y1 - cy) * ppu - a.shape[0] / 2)
            ya, yb, xa, xb = max(py, 0), min(py + a.shape[0], H), max(px, 0), min(px + a.shape[1], W)
            canvas[ya:yb, xa:xb] = np.maximum(canvas[ya:yb, xa:xb], d[ya - py:yb - py, xa - px:xb - px])
    r0, c0 = int((y0 + 1) * ppu), int(x0 * ppu)
    raw[r0:r0 + H, c0:c0 + W, 0] += canvas[::-1]
    return raw


def encode(rg):
    """r: density, sqrt-coded; g: reserve."""
    r = np.sqrt(np.clip(rg[..., 0], 0, DMAX) / DMAX)
    g = np.clip(rg[..., 1], 0, 1)
    return (np.stack([r, g], -1) * 255 + .5).astype("u1")


def bake_layer(gl, name, spec, noise, ktex, log=print, marks=None, cx0=None):
    P, ppu = spec["P"], spec["ppu"]
    W, H = int(round(P * ppu)), int(round(2 * ppu))
    A, B, bl = marks if marks is not None else PAINTERS[name]()
    t0 = time.time()
    state = [gl.target((W, H), clamp=False) for _ in range(2)]
    for st_ in state:
        st_.tex.repeat_y = False
    state[0].clear()
    canvas, origin = (P, 2.0), (0.0, -1.0)
    offs = [(-P, 0.0), (0.0, 0.0), (P, 0.0)]
    ink_tex.draw_marks(gl, state[0], A, canvas, origin, offs, noise=noise)
    n = bl.get("iters", 0)
    if n:
        gl.run("ink_kprep.frag", state[1], uS=state[0], uK=ktex, uCanvas=canvas, uOrigin=origin, uKTile=ink_tex.PAPER_P)
        px = (1 / W, 1 / H)
        for i in range(n):
            src, dst = state[(i + 1) % 2], state[i % 2]
            gl.run("ink_bleed.frag", dst, uS=src, uPx=px, uStep=float(1 + i % 3), uDt=bl["dt"])
        if n % 2 == 0:
            gl.run("blit.frag", state[0], uTex=state[1])
    ink_tex.draw_marks(gl, state[0], B, canvas, origin, offs, noise=noise)
    raw = np.frombuffer(state[0].tex.read(), "f2").reshape(H, W, 4)[..., :2].astype("f4")
    if name == "near" and marks is None:
        raw = inscribe(raw, ppu)
    out = encode(raw)
    for s in state:
        s.fbo.release()
        s.tex.release()
    log(f"  {name}: {W}x{H} verts={len(A.array()) + len(B.array())} {time.time() - t0:.1f}s")
    return out


def bake_sprite(gl, marks, x0, x1, y0, y1, ppu, noise, encode_rg=True):
    W, H = int((x1 - x0) * ppu), int((y1 - y0) * ppu)
    t = gl.target((W, H))
    t.clear()
    ink_tex.draw_marks(gl, t, marks, (x1 - x0, y1 - y0), (x0, y0), noise=noise)
    raw = np.frombuffer(t.tex.read(), "f2").reshape(H, W, 4)[..., :2].astype("f4")
    t.fbo.release()
    t.tex.release()
    return encode(raw)


def ensure(gl, log=print, rebake=()):
    """Load every baked asset from out/cache, baking whatever is missing. Returns a dict of arrays."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = source_key()
    d = CACHE / "ink"
    d.mkdir(parents=True, exist_ok=True)
    mat_file = d / "materials.npz"
    if mat_file.exists() and "materials" not in rebake:
        z = np.load(mat_file)
        mats = {k: z[k] for k in z.files}
    else:
        log("ink: baking materials")
        t0 = time.time()
        noise = (ink_tex.brush_noise() * 255 + .5).astype("u1")
        # the paper's 4096^2 fibre sheets come out differently in a context that has already drawn other things
        # (some driver state outlives the calls), so they are always laid down in a GL context of their own.
        # Creating it makes it current; leaving `with gl.ctx` hands the film's context back.
        with gl.ctx:
            own = GL()
            paper, K = ink_tex.bake_paper(own, ink_tex.tex2d(own, noise))
            own.ctx.release()
        mist = ink_tex.bake_mist()
        atlas, means = ink_tex.bake_atlas()
        mats = dict(noise=noise, paper=paper, K=K, mist=mist, atlas=atlas, atlas_mean=np.array(means, "f4"))
        np.savez(mat_file, **mats)
        log(f"  materials {time.time() - t0:.1f}s")
    out = dict(mats)
    ntex = ink_tex.tex2d(gl, mats["noise"])
    ktex = ink_tex.tex2d(gl, mats["K"])
    for name, spec in LAYERS.items():
        f = d / f"{name}.npy"
        kf = d / f"{name}.key"
        fresh = f.exists() and kf.exists() and (kf.read_text() == key or (rebake == "keep"))
        if rebake != "keep" and name in rebake:
            fresh = False
        if not fresh:
            log(f"ink: painting {name}")
            arr = bake_layer(gl, name, spec, ntex, ktex, log)
            np.save(f, arr)
            kf.write_text(key)
        out[name] = np.load(f, mmap_mode="r")
    f = d / "sprites.npz"
    kf = d / "sprites.key"
    if f.exists() and kf.exists() and (kf.read_text() == key or rebake == "keep") and not (rebake != "keep" and "sprites" in rebake):
        z = np.load(f)
        out["sun"], out["boat"] = z["sun"], z["boat"]
    else:
        s = SUN_SPRITE
        sun = bake_sprite(gl, paint_sun(), s["x0"], s["x1"], s["y0"], s["y1"], s["ppu"], ntex)
        b = BOAT_SPRITE
        boat = bake_sprite(gl, paint_boat(), -b["size"][0] / 2, b["size"][0] / 2, -b["size"][1] / 2, b["size"][1] / 2, b["ppu"], ntex)
        np.savez(f, sun=sun, boat=boat)
        kf.write_text(key)
        out["sun"], out["boat"] = sun, boat
    ntex.release()
    ktex.release()
    return out
