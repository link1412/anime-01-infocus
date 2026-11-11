"""The room's light, computed once.

A GPU path tracer gathers the light arriving at every visible surface of the room, stored in an atlas of orthographic
cards (each object is photographed from the axes the camera can see it from). The bake keeps separate passes: the screen
as four bilinear corners at unit white, and the night window. Any frame's lighting is a weighted sum of those passes.
"""
from __future__ import annotations
import hashlib, pathlib, time
import numpy as np
import moderngl
from scipy import ndimage
from ..core import GL, SHADERS

G = dict(ROOM=0, DESK=1, MON=2, STAND=3, MACH=4, KB=5, MUG=6, PAPER=7, SPK=8, CURT=9, SHELF=10, CABLE=11, PLATE=12)
AXES = dict(x=0, y=1, z=2)
R0, R1 = (-1.05, -.952, -.40), (1.35, 1.38, 2.00)

# group, axis, sign, (u0, v0, u1, v1), plane, length, metres per texel. u,v are the other two axes, in x,y,z order.
CARDS = [
    # the room's own planes: only the back wall is ever looked at; the rest is for what the gloss sees
    ("ROOM", "z", 1, (-1.05, -.952, 1.35, 1.38), R0[2], 0, .0060),
    ("ROOM", "z", -1, (-1.05, -.952, 1.35, 1.38), R1[2], 0, .0240),
    ("ROOM", "x", 1, (-.952, -.40, 1.38, 2.00), R0[0], 0, .0240),
    ("ROOM", "x", -1, (-.952, -.40, 1.38, 2.00), R1[0], 0, .0240),
    ("ROOM", "y", 1, (-1.05, -.40, 1.35, 2.00), R0[1], 0, .0240),
    ("ROOM", "y", -1, (-1.05, -.40, 1.35, 2.00), R1[1], 0, .0240),
    ("DESK", "y", 1, (-.81, -.31, .81, .47), -.228, .02, .0014),
    ("MON", "z", 1, (-.312, -.187, .312, .181), .012, .03, .0010),
    ("STAND", "z", 1, (-.125, -.236, .125, -.03), .08, .24, .0011),
    ("STAND", "y", 1, (-.125, -.18, .125, .07), -.215, .03, .0011),
    ("MACH", "z", 1, (.222, -.235, .378, -.178), -.045, .03, .0009),
    ("MACH", "y", 1, (.222, -.205, .378, -.045), -.176, .03, .0009),
    ("MACH", "x", -1, (-.235, -.205, -.178, -.045), .222, .03, .0009),
    ("KB", "y", 1, (-.31, .10, .035, .27), -.18, .06, .00075),
    ("KB", "z", 1, (-.31, -.236, .035, -.185), .27, .05, .0009),
    ("KB", "x", 1, (-.236, .10, -.185, .27), .035, .05, .0009),
    ("MUG", "z", 1, (.12, -.236, .245, -.14), .428, .1, .0006),
    ("MUG", "y", 1, (.12, .328, .245, .428), -.13, .1, .0006),
    ("MUG", "x", -1, (-.236, .328, -.14, .428), .12, .1, .0006),
    ("MUG", "x", 1, (-.236, .328, -.14, .428), .245, .1, .0006),
    ("PAPER", "y", 1, (-.44, -.13, -.08, .21), -.20, .05, .0009),
    ("SPK", "z", 1, (-.48, -.236, -.325, -.02), -.09, .17, .0010),
    ("SPK", "x", 1, (-.236, -.26, -.02, -.09), -.325, .15, .0012),
    ("SPK", "y", 1, (-.48, -.26, -.325, -.09), -.018, .05, .0012),
    ("CURT", "z", 1, (.36, -.95, 1.34, 1.37), -.30, .10, .0040),
    ("SHELF", "y", -1, (-.70, -.41, .57, -.21), .22, .07, .0022),
    ("SHELF", "z", 1, (-.70, .25, .57, .49), -.21, .15, .0018),
    ("CABLE", "z", 1, (-.34, -.34, .36, -.09), -.04, .30, .0012),
    ("CABLE", "y", 1, (-.34, -.35, .36, -.04), -.09, .30, .0012),
    ("PLATE", "z", 1, (-.52, .005, -.415, .11), -.39, .03, .0015),
]
PAD = 3
ATLAS_W = 2048
SPP = 8            # per fragment, per draw
TILE = 256


def _layout():
    """shelf-pack the cards. returns [(x, y, w, h)] in texels."""
    sizes = []
    for c in CARDS:
        u0, v0, u1, v1 = c[3]
        sizes.append((int(np.ceil((u1 - u0) / c[6])), int(np.ceil((v1 - v0) / c[6]))))
    order = sorted(range(len(CARDS)), key=lambda i: -sizes[i][1])
    x = y = rowh = 0
    rects = [None] * len(CARDS)
    for i in order:
        w, h = sizes[i]
        if x + w + 2 * PAD > ATLAS_W:
            x = 0
            y += rowh
            rowh = 0
        rects[i] = (x + PAD, y + PAD, w, h)
        x += w + 2 * PAD
        rowh = max(rowh, h + 2 * PAD)
    H = y + rowh
    return rects, (ATLAS_W, int(np.ceil(H / 64) * 64))


def card_tables(rects, size):
    """what the scene shader needs to find a card: rect in atlas uv, domain, plane; and a group/axis -> card index table."""
    W, H = size
    rect, dom, plane = [], [], []
    of = [-1] * (16 * 6)
    for i, c in enumerate(CARDS):
        x, y, w, h = rects[i]
        rect.append((x / W, y / H, w / W, h / H))
        u0, v0, u1, v1 = c[3]
        dom.append((u0, v0, u1 - u0, v1 - v0))
        plane.append((c[4], float(c[2]), .5 / w, .5 / h))
        of[G[c[0]] * 6 + AXES[c[1]] * 2 + (0 if c[2] > 0 else 1)] = i
    return rect, dom, plane, of


def bake_key(batches):
    h = hashlib.sha1(str(batches).encode())
    for f in ("desk_geo.glsl", "desk_bake_gbuf.frag", "desk_bake_pt.frag"):
        h.update((SHADERS / f).read_bytes())
    h.update(repr((CARDS, PAD, ATLAS_W)).encode())
    return h.hexdigest()[:12]


def _mrt(ctx, size, n):
    tex = [ctx.texture(size, 4, dtype="f4") for _ in range(n)]
    for t in tex:
        t.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return ctx.framebuffer(tex), tex


class _Fb:
    def __init__(self, fbo):
        self.fbo = fbo


def _read(tex, size):
    return np.frombuffer(tex.read(), "f4").reshape(size[1], size[0], 4).copy()


def run_bake(gl: GL, batches, log=print):
    ctx = gl.ctx
    rects, size = _layout()
    W, H = size
    log(f"[desk bake] {len(CARDS)} cards, atlas {W}x{H}, {sum(r[2] * r[3] for r in rects) / 1e6:.2f} Mtexel")
    # where the surfaces are
    fb, (tP, tN) = _mrt(ctx, size, 2)
    fb.clear(0, 0, 0, 0)
    shim = _Fb(fb)
    for c, (x, y, w, h) in zip(CARDS, rects):
        u0, v0, u1, v1 = c[3]
        fb.viewport = (x, y, w, h)
        fb.scissor = (x, y, w, h)
        gl.run("desk_bake_gbuf.frag", shim, uAxis=AXES[c[1]], uSign=float(c[2]), uPlane=float(c[4]), uLen=float(c[5]),
               uIsRoom=1.0 if c[0] == "ROOM" else 0.0, uGroup=float(G[c[0]]), uDom=(u0, v0, u1, v1),
               uMask=(1 << G[c[0]]))
        ctx.finish()
    fb.viewport = (0, 0, W, H)
    fb.scissor = None
    gp, gn = _read(tP, size), _read(tN, size)
    valid = gp[..., 3] > .5
    log(f"[desk bake] {valid.sum() / 1e6:.2f} Mtexel on surfaces")
    # gather light, in tiles, so that the GPU's watchdog never sees a long draw
    tiles = []
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            if valid[ty:ty + TILE, tx:tx + TILE].any():
                tiles.append((tx, ty, min(TILE, W - tx), min(TILE, H - ty)))
    out, tO = _mrt(ctx, size, 5)
    shimO = _Fb(out)
    acc = [np.zeros((H, W, 3), "f8") for _ in range(5)]
    t0 = time.time()
    for b in range(batches):
        out.scissor = None
        out.clear(0, 0, 0, 0)
        for (tx, ty, tw, th) in tiles:
            out.scissor = (tx, ty, tw, th)
            gl.run("desk_bake_pt.frag", shimO, uGP=tP, uGN=tN, uBatch=float(b), uSpp=SPP, uMask=8191)
            ctx.finish()
        out.scissor = None
        for i in range(5):
            acc[i] += _read(tO[i], size)[..., :3]
        if b % 8 == 0 or b == batches - 1:
            el = time.time() - t0
            log(f"[desk bake] {(b + 1) * SPP} spp  {el:.0f}s  eta {el / (b + 1) * (batches - b - 1):.0f}s")
    acc = [a / batches for a in acc]
    # what few stray paths left behind
    for i in range(5):
        lum = acc[i].mean(-1)
        med = ndimage.median_filter(lum, size=3)
        k = np.where(lum > 2.5 * med + 1e-5, (2.5 * med + 1e-5) / np.maximum(lum, 1e-9), 1.0)
        acc[i] *= k[..., None]
    # fill what lies off the surfaces inside each card, so that bilinear lookups never touch nothing
    atlas = [np.zeros((H, W, 4), "f4") for _ in range(5)]
    for c, (x, y, w, h) in zip(CARDS, rects):
        X0, Y0, X1, Y1 = x - PAD, y - PAD, x + w + PAD, y + h + PAD
        m = valid[Y0:Y1, X0:X1]
        if not m.any():
            continue
        idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
        # the room's planes only ever carry slow light: what's left of the noise can go
        sigma = 3.0 if c[0] in ("ROOM", "CURT") else .7
        for i in range(5):
            v = acc[i][Y0:Y1, X0:X1]
            w_ = ndimage.gaussian_filter(m.astype("f8"), sigma)
            v = ndimage.gaussian_filter(v * m[..., None], (sigma, sigma, 0)) / np.maximum(w_[..., None], 1e-3)
            atlas[i][Y0:Y1, X0:X1, :3] = v[idx[0], idx[1]]
        atlas[0][Y0:Y1, X0:X1, 3] = gn[Y0:Y1, X0:X1, 3][idx[0], idx[1]]
    tP.release(); tN.release(); fb.release()
    for t in tO:
        t.release()
    out.release()
    return atlas, rects, size


class Baked:
    """the atlas, on the GPU, and how to find things in it."""

    def __init__(self, gl: GL, cache_dir="out/cache", batches=192):
        key = bake_key(batches)
        path = pathlib.Path(cache_dir) / f"desk_light_{key}.npz"
        rects, size = _layout()
        if path.exists():
            d = np.load(path)
            atlas = [d[f"a{i}"] for i in range(5)]
        else:
            t0 = time.time()
            atlas, rects, size = run_bake(gl, batches)
            print(f"[desk bake] done in {time.time() - t0:.0f}s", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **{f"a{i}": atlas[i].astype("f2") for i in range(5)})
            tmp.rename(path)
        self.tex = []
        for a in atlas:
            t = gl.ctx.texture((size[0], size[1]), 4, a.astype("f2").tobytes(), dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.tex.append(t)
        self.rect, self.dom, self.plane, self.of = card_tables(rects, size)
        pad = [(0.0, 0.0, 0.0, 0.0)] * (40 - len(self.rect))
        self.rect, self.dom, self.plane = self.rect + pad, self.dom + pad, self.plane + pad
        self.uniforms = dict(uA0=self.tex[0], uA1=self.tex[1], uA2=self.tex[2], uA3=self.tex[3], uAW=self.tex[4],
                             uCardRect=self.rect, uCardDom=self.dom, uCardPlane=self.plane, uCardOf=self.of)
