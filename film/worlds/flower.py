"""Rain, a low sun, and a poppy at thirty centimetres.

The light is integrated per texel of each petal (texture space), through two layers of petal,
because a petal is a filter and not a wall. The lens is integrated by rendering the flower once
for every sample of the aperture. The far side is the sea world itself, convolved with the
shape of the aperture. Nothing here is a photograph or was ever near one.
"""
from __future__ import annotations
import numpy as np
import moderngl
from scipy import fft as sf
from ..core import GL, Target
from . import flower_geo as geo
from .flower_drop import HangingDrop, FAR_YAW, FAR_PITCH
from .flower_arrival import Arrival, T_LAND, T_OVER

K_COC = .096
TAN = .129              # 1 view unit = 4 cm at the hero's distance
A_LENS = K_COC * TAN    # radius of the entrance pupil, metres
M_LIGHTS = 32
TILE = 768
ATLAS = 6 * TILE


def vogel(n, rot=0.0):
    i = np.arange(n) + .5
    r = np.sqrt(i / n)
    th = i * 2.39996323 + rot
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def basis(d):
    d = np.asarray(d, "f4")
    a = np.array([0, 1, 0], "f4") if abs(d[1]) < .9 else np.array([1, 0, 0], "f4")
    x = np.cross(a, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return x, y


class Bokeh:
    """the far side of the frame through the actual aperture: overlap-add FFT, field dependent."""
    TX, TY = 5, 3

    def __init__(self, size):
        self.size = size                # half res (w,h)
        w, h = size
        self.win = []
        cx = (np.arange(self.TX) + .5) / self.TX * w
        cy = (np.arange(self.TY) + .5) / self.TY * h
        sx, sy = w / self.TX, h / self.TY
        xs, ys = np.arange(w) + .5, np.arange(h) + .5
        ww = [np.clip(1 - np.abs(xs - c) / sx, 0, 1) for c in cx]
        wh = [np.clip(1 - np.abs(ys - c) / sy, 0, 1) for c in cy]
        # edge tiles keep full weight out to the frame's edge
        ww[0][xs < cx[0]] = 1
        ww[-1][xs > cx[-1]] = 1
        wh[0][ys < cy[0]] = 1
        wh[-1][ys > cy[-1]] = 1
        for j in range(self.TY):
            for i in range(self.TX):
                x0, x1 = max(int(cx[i] - sx), 0), min(int(cx[i] + sx) + 1, w)
                y0, y1 = max(int(cy[j] - sy), 0), min(int(cy[j] + sy) + 1, h)
                W = np.outer(wh[j][y0:y1], ww[i][x0:x1]).astype("f4")
                f = ((cx[i] / w - .5) * 2 * (w / h), -(cy[j] / h - .5) * 2)
                self.win.append((x0, x1, y0, y1, W, f))

    @staticmethod
    def kernel(R, field, ch):
        """disc with a bright rim, a faint onion, cut by the rear of the barrel off axis."""
        loca = (.988, 1.012, .996)[ch]
        Rr = R * loca
        n = int(np.ceil(Rr)) + 3
        y, x = np.mgrid[-n:n + 1, -n:n + 1].astype("f4")
        d = np.hypot(x, y)
        a = np.clip(Rr - d + .5, 0, 1)
        fr = d / max(Rr, 1e-3)
        w = (.86 + .14 * fr ** 2 + .13 * np.clip((fr - .90) / .10, 0, 1)) * (1 + .010 * np.sin(fr * 34 + 1.2))
        # the cat's eye
        off = np.asarray(field, "f4") * .34 * Rr
        d2 = np.hypot(x + off[0], y - off[1])
        a2 = np.clip(Rr * 1.02 - d2 + .5, 0, 1)
        k = a * a2 * w
        return k / k.sum()

    def blur(self, img, R):
        if R < .6:
            return img
        h, w = img.shape[:2]
        out = np.zeros_like(img)
        for x0, x1, y0, y1, W, f in self.win:
            tile = img[y0:y1, x0:x1] * W[..., None]
            for c in range(3):
                k = self.kernel(R, f, c)
                kh, kw = k.shape
                H, Wd = (y1 - y0) + kh, (x1 - x0) + kw
                fh, fw = sf.next_fast_len(H, real=True), sf.next_fast_len(Wd, real=True)
                r = sf.irfft2(sf.rfft2(tile[..., c], s=(fh, fw), workers=-1) * sf.rfft2(k, s=(fh, fw), workers=-1), s=(fh, fw), workers=-1)
                oy, ox = kh // 2, kw // 2
                ya, yb = max(y0 - oy, 0), min(y1 - oy + kh, h)
                xa, xb = max(x0 - ox, 0), min(x1 - ox + kw, w)
                out[ya:yb, xa:xb, c] += r[ya - (y0 - oy):yb - (y0 - oy), xa - (x0 - ox):xb - (x0 - ox)]
        return out


class Group:
    """surfaces that share light and shadow."""

    def __init__(self, gl: GL, surfaces, lit_size, center, half, pivot, name):
        self.gl = gl
        self.name = name
        self.center = np.asarray(center, "f4")
        self.half = half
        self.pivot = tuple(float(v) for v in pivot)
        zs = np.concatenate([x["mesh"]["P"].reshape(-1, 3)[:, 2] for x in surfaces[:4]])
        self.zrange = (float(zs.min()), float(zs.max()))
        ctx = gl.ctx
        self.p_mesh = gl.prog("flower_pass.frag", "flower_mesh.vert")
        self.p_depth = gl.prog("flower_depth.frag", "flower_mesh.vert")
        self.p_tss = gl.prog("flower_tss.frag", "flower_mesh.vert")
        self.surf = []
        for s in surfaces:
            H, W = s["shape"][:2]
            vb = ctx.buffer(s["vb"].tobytes())
            ib = ctx.buffer(s["idx"].tobytes())
            lo = ctx.buffer(s["lo"].tobytes())
            fmt = "3f 3f 3f 3f 2f 1f"
            names = ["in_p", "in_n", "in_t", "in_b", "in_uv", "in_w"]
            pig = self._tex(s["pig"])
            det = self._tex(s["det"])
            size = lit_size if s["pig"].shape[0] >= 1024 else min(s["pig"].shape[0], lit_size)
            lit = gl.target((size, size), mips=True)
            self.surf.append(dict(pig=pig, det=det, lit=lit, vb=vb, ib=ib,
                                  a=ctx.vertex_array(self.p_mesh, [(vb, fmt, *names)], ib, skip_errors=True),
                                  t=ctx.vertex_array(self.p_tss, [(vb, fmt, *names)], ib, skip_errors=True),
                                  d=ctx.vertex_array(self.p_depth, [(vb, fmt, *names)], lo, skip_errors=True),
                                  ))

    def _tex(self, a):
        h, w = a.shape[:2]
        t = self.gl.ctx.texture((w, h), 4, a.astype("f2").tobytes(), dtype="f2")
        t.build_mipmaps()
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        t.anisotropy = 8.0
        return t


class Flower:
    N_PASS = 128
    IMPACTS = [(.62, 2, .0011), (2.1, 3, -.0019), (3.35, 0, -.0009), (4.2, 2, -.0012), (5.45, 3, .0014), (6.7, 1, -.0009), (8.2, 3, -.0010), (T_LAND, 1, -.0009), (9.6, 0, -.0011), (T_OVER + .08, 1, -.0006), (10.7, 2, -.0008), (11.1, 1, -.0004), (13.3, 1, -.0004)]
    T_RUN = (2.1, 5.45)
    FAR_MEDIAN = .055       # whatever exposure the beach is kept at in its own shot, behind this flower it is low key
    FAR_MAX = 4.0           # and what the sun does on wet sand can't be allowed to flood a whole lens
    ENV_MEDIAN = .46
    N_BEAD = 320

    def __init__(self, gl: GL, size, sea=None):
        self.gl = gl
        self.size = tuple(size)
        self.aspect = size[0] / size[1]
        self.sea = sea
        ctx = gl.ctx
        data = geo.build()
        self.data = data
        self.hero = Group(gl, data["hero"], 1536, np.array(geo.HERO_POS) + np.array([0, .012, -.004]), .075, np.array(geo.HERO_POS) + np.array([.003, -.09, .02]), "hero")
        self.second = Group(gl, data["second"], 1024, np.array(geo.SECOND_POS) + np.array([0, .012, -.004]), .075, np.array(geo.SECOND_POS) + np.array([-.01, -.09, .02]), "second")
        # shadow atlases
        self.z1 = gl.target((ATLAS, ATLAS), comps=1, dtype="f4", nearest=True)
        self.z2 = gl.target((ATLAS, ATLAS), comps=1, dtype="f4", nearest=True)
        self.z_tmp = self.z1
        self.dep = ctx.depth_renderbuffer((ATLAS, ATLAS))
        self.fb1 = ctx.framebuffer([self.z1.tex], self.dep)
        self.fb2 = ctx.framebuffer([self.z2.tex], self.dep)
        self.z1s = gl.target((ATLAS // 2, ATLAS // 2), comps=1, dtype="f4", nearest=True)
        self.z2s = gl.target((ATLAS // 2, ATLAS // 2), comps=1, dtype="f4", nearest=True)
        self.deps = ctx.depth_renderbuffer((ATLAS // 2, ATLAS // 2))
        self.fb1s = ctx.framebuffer([self.z1s.tex], self.deps)
        self.fb2s = ctx.framebuffer([self.z2s.tex], self.deps)
        # lens pass
        self.t_pass = gl.target(size)
        self.d_pass = ctx.depth_renderbuffer(self.size)
        self.fb_pass = ctx.framebuffer([self.t_pass.tex], self.d_pass)
        self.t_hero = gl.target(size, dtype="f4")
        self.t_second = gl.target(size, dtype="f4")
        self.t_tmp = gl.target(size, dtype="f4")
        self.t_beads = gl.target(size, dtype="f4")
        self.t_beads2 = gl.target(size, dtype="f4")
        # far
        self.half = (size[0] // 2, size[1] // 2)
        self.t_sea = gl.target(size)
        self.t_seah = gl.target(self.half)
        self.t_far = gl.target(self.half)
        self.bokeh = Bokeh(self.half)
        self._far_k = None
        self._env = None
        # beads
        self.p_bead = gl.prog("flower_bead.frag", "flower_bead.vert")
        self.bead_vaos = {}
        self.bead_vbs, self.bead_arr = {}, {}
        self.drop = HangingDrop(self, TAN, K_COC, A_LENS)
        a0, a1 = np.array(self.drop.view_apex), np.array(self.drop.view_q)
        for name, arr in (("beads", data["beads"]), ("micro", data["micro"])):
            # nothing sits on the drop or shows through it: whatever is behind it, it has already bent
            z = arr[:, 2]
            v = np.stack([arr[:, 0] / z / TAN, arr[:, 1] / z / TAN], 1) - a0
            ab = a1 - a0
            s = np.clip(v @ ab / (ab @ ab), 0, 1)
            d = np.hypot(*(v - s[:, None] * ab).T)
            coc = K_COC * np.abs(1 / self.drop.z - 1 / z)
            near = np.hypot(*(v - ab).T)         # and where it joins the petal the water has already gathered them all
            arr = arr[(d > self.drop.rv * 1.05 + arr[:, 12] * 1.3 / z / TAN + coc) & (near > self.drop.rv * 1.7 + coc)]
            for ip in range(4):
                sel = arr[arr[:, 14] == ip]
                if len(sel) == 0:
                    continue
                inst = self._bead_inst(sel)
                vb = ctx.buffer(inst.tobytes())
                vao = gl.instanced(self.p_bead, "3f 3f 3f 3f 4f 4f 2f/i", ["ic", "inb", "it", "ib", "ishape", "iuv", "iw"], vb)
                self.bead_vaos[(name, ip)] = (vao, len(sel))
                self.bead_vbs[(name, ip)], self.bead_arr[(name, ip)] = vb, sel
        arr = data["beads2"]
        for ip in range(4):
            sel = arr[arr[:, 14] == ip]
            if len(sel):
                vb = ctx.buffer(self._bead_inst(sel).tobytes())
                self.bead_vaos[("beads2", ip)] = (gl.instanced(self.p_bead, "3f 3f 3f 3f 4f 4f 2f/i", ["ic", "inb", "it", "ib", "ishape", "iuv", "iw"], vb), len(sel))
        self._lights()
        self._calm = 1.0
        self._runner_setup()
        self.arrival = Arrival(self)
        self.arr_buf = ctx.buffer(reserve=24 * 4, dynamic=True)
        self.v_arr = gl.instanced(self.p_bead, "3f 3f 3f 3f 4f 4f 2f/i", ["ic", "inb", "it", "ib", "ishape", "iuv", "iw"], self.arr_buf)
        self.v_trail = gl.instanced(self.p_bead, "3f 3f 3f 3f 4f 4f 2f/i", ["ic", "inb", "it", "ib", "ishape", "iuv", "iw"], ctx.buffer(self.arrival.trail.tobytes()))
        self.p_streak = gl.prog("flower_streak.frag", "flower_streak.vert")
        self.streak_vb = ctx.buffer(reserve=7 * 4 * 64, dynamic=True)
        self.v_streak = gl.instanced(self.p_streak, "4f 3f/i", ["iseg", "ibody"], self.streak_vb)

    @staticmethod
    def _bead_inst(a):
        # P n t b | R e ip u v dudm dvdm w zc
        R, e, u, v, dudm, dvdm, w, zc = a[:, 12], a[:, 13], a[:, 15], a[:, 16], a[:, 17], a[:, 18], a[:, 19], a[:, 20]
        return np.column_stack([a[:, :3], a[:, 3:6], a[:, 6:9], a[:, 9:12], R, e, np.zeros(len(a)), zc, u, v, dudm, dvdm, w, np.zeros(len(a))]).astype("f4")

    # ---- the sky as a set of lights ---------------------------------------------------------
    def _lights(self):
        Lsun = np.array([1.35 * TAN, 1.25 * TAN, 1.0], "f4")      # toward the sun: just out of frame, top right
        Lsun /= np.linalg.norm(Lsun)
        self.Lsun = Lsun
        ds, es = [], []
        xs, ys = basis(Lsun)
        S = 16
        pts = vogel(S) * .10        # a hazy sun: a few degrees across
        E_sun = np.array([1.0, .78, .52], "f4") * 3.4
        for p in pts:
            d = -(Lsun + xs * p[0] + ys * p[1])
            ds.append(d / np.linalg.norm(d))
            es.append(E_sun / S)
        # the sky: bright low down toward the sun, grey-blue overhead and behind
        n = M_LIGHTS - S
        for i in range(n):
            y = (i + .5) / n
            th = i * 2.39996323
            r = np.sqrt(1 - y * y)
            w = np.array([r * np.sin(th), y, r * np.cos(th)], "f4")
            toward = max(float(w @ Lsun), 0.0)
            L = np.array([.20, .25, .36], "f4") * .40 * (.65 + .35 * (1 - y)) + np.array([1.0, .70, .46], "f4") * .40 * toward ** 3
            es.append(L * (2 * np.pi / n))
            ds.append(-w)
        self.Ld = np.array(ds, "f4")
        self.Le = np.array(es, "f4")
        self.Lx = np.zeros_like(self.Ld)
        self.Ly = np.zeros_like(self.Ld)
        for i, d in enumerate(self.Ld):
            self.Lx[i], self.Ly[i] = basis(d)
        cols = 6
        self.tiles = [(i % cols, i // cols) for i in range(M_LIGHTS)]

    def _sway(self, t, group):
        """the wind and the rain: rotations small enough to be a matrix of small angles."""
        s = (1.0 if group == "hero" else .85) / self._calm
        ph = 0.0 if group == "hero" else 2.2
        ax = s * (.0065 * np.sin(t * 1.13 + ph) + .0035 * np.sin(t * 2.71 + 1.0 + ph) + .0014 * np.sin(t * 6.3 + ph))
        az = s * (.0052 * np.sin(t * .87 + 2.0 + ph) + .0031 * np.sin(t * 2.23 + ph))
        c1, s1 = np.cos(ax), np.sin(ax)
        c2, s2 = np.cos(az), np.sin(az)
        Rx = np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]], "f4")
        Rz = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]], "f4")
        return tuple(float(v) for v in (Rz @ Rx).T.ravel())

    def _flut(self, t, i):
        f = float(.00042 * np.sin(t * (2.3 + .37 * i) + i * 1.9) + .00021 * np.sin(t * (5.1 + .6 * i) + i * 4.1)) / self._calm
        # rain lands on it. a petal takes the hit and rings
        for t0, k, a in self.IMPACTS:
            if k == i and t > t0:
                u = t - t0
                f += a * np.exp(-3.4 * u) * np.sin(21 * u) / self._calm
        return f

    # ---- light ------------------------------------------------------------------------------
    def _shadows(self, G, t, sway, fb1, fb2, z1, tile, cam):
        gl, ctx = self.gl, self.gl.ctx
        common = dict(uSway=sway, uPivot=G.pivot, uSpace=2, uLo=tuple(float(v) for v in G.center), uLs=(G.half, G.half), **cam)
        fb1.use()
        fb1.clear(1e3, 0, 0, 0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        for peel, fb in ((0.0, fb1), (1.0, fb2)):
            fb.use()
            if peel > .5:
                fb.clear(1e3, 0, 0, 0, depth=1.0)
            for i in range(M_LIGHTS):
                cx, cy = self.tiles[i]
                fb.viewport = (cx * tile, cy * tile, tile, tile)
                fb.scissor = (cx * tile, cy * tile, tile, tile)
                lm = np.stack([self.Lx[i], self.Ly[i], self.Ld[i]], 0)      # rows
                for k, s in enumerate(G.surf):
                    gl.bind(self.p_depth_any(G), dict(common, uLm=tuple(float(v) for v in lm.T.ravel()), uPig=s["pig"], uZ1=z1.tex, uPeel=peel, uBias=.0009,
                                                      uFlut=self._flut(t, k) if k < 4 else 0.0))
                    s["d"].render(moderngl.TRIANGLES)
            ctx.finish()
            fb.scissor = None
            fb.viewport = (0, 0, *fb.size)
        ctx.disable(moderngl.DEPTH_TEST)

    @staticmethod
    def p_depth_any(G):
        return G.p_depth

    def _tss(self, G, t, sway, z1, z2, tile_size, atlas, cam, fill):
        gl, ctx = self.gl, self.gl.ctx
        ti = [(c * tile_size / atlas, r * tile_size / atlas, tile_size / atlas, tile_size / atlas) for c, r in self.tiles]
        base = dict(uSway=sway, uPivot=G.pivot, uSpace=1,
                    uLd=[tuple(float(v) for v in x) for x in self.Ld], uLe=[tuple(float(v) for v in x) for x in self.Le],
                    uLx=[tuple(float(v) for v in x) for x in self.Lx], uLy=[tuple(float(v) for v in x) for x in self.Ly],
                    uLo=tuple(float(v) for v in G.center), uLs=(G.half, G.half), uTile=ti, uTocc=(.80, .46, .12),
                    uFill=fill, uTrans=.74, uRefl=.50, uKDet=.8, uForward=1.3, **cam)
        for k, s in enumerate(G.surf):
            s["lit"].fbo.use()
            s["lit"].clear()
            u = dict(base, uPig=s["pig"], uDet=s["det"], uZ1=z1.tex, uZ2=z2.tex, uFlut=self._flut(t, k) if k < 4 else 0.0)
            if k >= 4:      # what stands in the middle of the cup, in the light of all of it
                u["uFill"] = (fill[0] * .92, fill[1] * 2.2, fill[2] * 8.0)
            gl.bind(G.p_tss, u)
            s["t"].render(moderngl.TRIANGLES)
            s["lit"].gen()
        ctx.finish()

    # ---- the far side ------------------------------------------------------------------------
    def _far(self, st, zoom, fd, gc):
        gl = self.gl
        t = st["t"]
        if zoom > 16:       # the bead fills the frame; what is around it can't be seen
            return
        if self.sea is not None:
            # the camera looks at the place the sun isn't, down onto the sand; the sun is where the light says
            yaw, pitch = float(np.arctan(gc[0] * TAN)) + FAR_YAW, float(np.arctan(gc[1] * TAN)) + FAR_PITCH
            self.sea.render(dict(t=t, yaw=yaw, pitch=pitch, roll=0.0, tan_half=TAN / zoom, yaw_v=0.0, drop=None, crack=None), self.t_sea)
        else:
            self.t_sea.clear(0, 0, 0, 1)
        gl.run("blit.frag", self.t_seah, uTex=self.t_sea)
        a = np.frombuffer(self.t_seah.fbo.read(components=3, dtype="f4"), "f4").reshape(self.half[1], self.half[0], 3).astype("f4").copy()
        if self.sea is None:
            a = self._stub()
        else:
            if self._far_k is None:
                # set once, from the first look at it, so that it can't breathe
                self._far_k = self.FAR_MEDIAN / max(float(np.median(a.mean(-1))), 1e-3)
                print(f"[flower] far plate gain {self._far_k:.4f}", flush=True)
            a = np.minimum(a * self._far_k, self.FAR_MAX)
        R = min(K_COC * fd * zoom * self.size[1] / 2 / 2, 150.0)
        b = self.bokeh.blur(a, R)
        self.t_far.tex.write(np.concatenate([b, np.ones((*b.shape[:2], 1), "f4")], -1).astype("f2").tobytes())

    def _scaled_env(self):
        """the sky as this flower sees it: the same sky, on this shot's exposure."""
        if self.sea is None:
            return self._stub_env()
        if self._env is None:
            src = self.sea.sky.env
            w, h = src.tex.size
            a = np.frombuffer(src.fbo.read(components=3, dtype="f4"), "f4")
            k = self.ENV_MEDIAN / max(float(np.median(a)), 1e-3)
            print(f"[flower] env gain {k:.4f}", flush=True)
            self._env = self.gl.target((w, h), mips=True, clamp=False)
            self._env.tex.repeat_y = False
            self.gl.run("flower_add.frag", self._env, uTex=src, uW=float(k))
            self._env.gen()
        return self._env

    def _stub_env(self):
        if not hasattr(self, "_senv"):
            h, w = 128, 512
            el = (np.arange(h) + .5) / h
            az = ((np.arange(w) + .5) / w - .5) * 6.283
            Az, El = np.meshgrid(az, el)
            c = np.array([.42, .46, .56], "f4")[None, None] * (.55 + .45 * np.exp(-El[..., None] / .35))
            g = np.exp(-(Az[..., None] / .9) ** 2) * np.exp(-El[..., None] / .15)
            c = c + np.array([2.2, 1.5, .9], "f4")[None, None] * g
            self._senv = self.gl.target((w, h), mips=True, clamp=False)
            self._senv.tex.write(np.concatenate([c, np.ones((h, w, 1))], -1).astype("f2").tobytes())
            self._senv.gen()
        return self._senv

    def _stub(self):
        if not hasattr(self, "_stub_a"):
            h, w = self.half[1], self.half[0]
            y = ((np.arange(h) + .5) / h - .5)[:, None] * 2
            hz = .74
            sky = np.array([.62, .50, .40], "f4") * np.exp(-np.maximum(y - hz, 0)[..., None] / .5)
            sea = np.array([.040, .046, .054], "f4") * (1 + 2.2 * np.exp(-(hz - y)[..., None].clip(0) / .05)) * (.5 + .5 * np.exp(-(hz - y)[..., None].clip(0) / .5))
            a = np.where((y > hz)[..., None], sky, sea).repeat(w, 1).astype("f4").reshape(h, w, 3)
            rng = np.random.default_rng(3)
            for _ in range(420):
                px, py = rng.integers(0, w), int(h - 1 - (rng.uniform(.04, .55) ** 1.2 * h + .14 * h))
                a[py, px] += np.array([1.0, .86, .7], "f4") * float(np.exp(rng.normal(np.log(14), 1.0)))
            self._stub_a = a
        return self._stub_a

    # ---- a frame ------------------------------------------------------------------------------
    def render(self, st: dict, out: Target):
        gl = self.gl
        t, fd = st["t"], st["fd"]
        z0 = st.get("zoom", 1.0)
        self._calm = z0
        # the frame is all drop, and the drop has come to rest on the sea
        if self.sea is not None and st.get("morph", self.drop.morph(z0)) >= .999:
            end = st.get("end", (0.0, -.095, .27))
            self.sea.render(dict(t=t, yaw=end[0], pitch=end[1], roll=np.pi, tan_half=end[2], yaw_v=0.0, drop=None, crack=None), out)
            return
        zoom = z0 * (1 + .0035 * fd / z0)
        gc = st.get("gc", (0.0, 0.0))
        cam = dict(uC=(0.0, 0.0, 0.0), uF=(0.0, 0.0, 1.0), uR=(1.0, 0.0, 0.0), uU=(0.0, 1.0, 0.0))
        sway_h, sway_s = self._sway(t, "hero"), self._sway(t, "second")
        self._far(st, zoom, fd, gc)
        # light, once per frame
        camm = dict(uC=cam["uC"], uF=cam["uF"], uR=cam["uR"], uU=cam["uU"], uTan=TAN, uFd=fd, uLens=(0.0, 0.0), uGc=(0.0, 0.0), uZoom=1.0, uAspect=self.aspect, uJit=(0.0, 0.0))
        self._shadows(self.hero, t, sway_h, self.fb1, self.fb2, self.z1, TILE, camm)
        self._tss(self.hero, t, sway_h, self.z1, self.z2, TILE, ATLAS, camm, (.44, .18, .018))
        self._shadows(self.second, t, sway_s, self.fb1s, self.fb2s, self.z1s, TILE // 2, camm)
        self._tss(self.second, t, sway_s, self.z1s, self.z2s, TILE // 2, ATLAS // 2, camm, (.36, .26, .05))
        # lens
        N = int(st.get("passes", self.N_PASS))
        env = self._scaled_env()
        self._lens(self.second, sway_s, t, fd, zoom, gc, cam, env, max(N * 3 // 4, 16), "beads2", self.t_second, self.t_tmp, 9.0, 1)
        self._lens(self.hero, sway_h, t, fd, zoom, gc, cam, env, N, "beads", self.t_hero, self.t_tmp, 0.0, 2)
        self._beads(st, t, fd, zoom, gc, cam, env, sway_h, sway_s)
        if st.get("no_drop"):
            self.drop.t_acc.clear()
        else:
            self.drop.render(st, t, fd, zoom, gc, cam, env, sway_h, sway_s)
        gl.run("flower_comp.frag", out, uHero=self.t_hero, uSecond=self.t_second, uFar=self.t_far, uBeads=self.t_beads, uBeads2=self.t_beads2, uDrop=self.drop.t_acc)
        self._streaks(out, t, fd, zoom, gc)

    def snapshot(self, t, gc, fd_near, fd_far):
        """the picture this flower makes, as arrays (rows bottom first), for the world in which it gets written down."""
        def rd(tg, n=4):
            h, w = tg.tex.size[1], tg.tex.size[0]
            return np.frombuffer(tg.fbo.read(components=n, dtype="f4"), "f4").reshape(h, w, n).copy()
        st = dict(t=t, zoom=1.0, gc=gc, passes=64, bead_passes=16, no_drop=True)
        o = self.gl.target(self.size)
        self.render(dict(st, fd=fd_near), o)
        hero = rd(self.t_hero)
        far = rd(self.t_sea, 3)
        far = np.minimum(far * self._far_k, self.FAR_MAX * 2) if self.sea is not None else far
        self.render(dict(st, fd=fd_far), o)
        second = rd(self.t_second)
        self.gl.ctx.finish()
        return dict(hero=hero, second=second, far=far, gc=tuple(gc))

    def _streaks(self, out, t, fd, zoom, gc):
        """drops crossing during 1/250 s. at this range one drop lasts one frame."""
        if zoom > 3:
            return
        gl, ctx = self.gl, self.gl.ctx
        rng = np.random.default_rng(int(t * 30 + .5) + 77)
        n = rng.integers(2, 7)
        inst = []
        # the ones the camera has been waiting for
        frame = int(t * 30 + .5)
        P = self.runner(self.T_RUN[0] + 1e-3)["P"] if frame == int(self.T_RUN[0] * 30 + .5) - 1 else self.arrival.at(T_LAND + 1e-3)["P"] if frame == int(T_LAND * 30 + .5) - 1 else None
        if P is not None:
            z = float(P[2])
            vx, vy = float(P[0] / z / TAN), float(P[1] / z / TAN)
            ln = 6.0 / 250 / (z * TAN)
            x0, y0 = (vx - gc[0]) * zoom, (vy - gc[1]) * zoom
            inst.append([(x0 - ln * zoom * .06) / self.aspect, y0 + ln * zoom, x0 / self.aspect, y0, .014 * zoom, .13, 1.0])
        for _ in range(n):
            z = float(np.exp(rng.uniform(np.log(.12), np.log(2.2))))
            x, y = rng.uniform(-2, 2) * self.aspect / 1.78, rng.uniform(-1.3, 1.5)
            ln = 6.0 / 250 / (z * TAN)                     # view units
            coc = K_COC * abs(fd - 1 / z) * zoom
            hw = max(.0011 / (z * TAN), coc)
            bright = .075 * (.0011 / (z * TAN)) / hw * rng.uniform(.5, 1.2)
            x0, y0 = (x - gc[0]) * zoom, (y - gc[1]) * zoom
            ax, ay = x0 / self.aspect, y0
            bx, by = (x0 + ln * zoom * .06) / self.aspect, y0 - ln * zoom
            inst.append([ax, ay, bx, by, hw * zoom, bright, .9])
        self.streak_vb.orphan()
        self.streak_vb.write(np.array(inst, "f4").tobytes())
        out.fbo.use()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        gl.bind(self.p_streak, dict(uAspect=self.aspect, uTint=(1.0, .88, .74)))
        self.v_streak.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(inst))
        ctx.disable(moderngl.BLEND)

    def _beads(self, st, t, fd, zoom, gc, cam, env, sway_h, sway_s):
        """water, with many more samples of the lens than the petals get: a blurred bright drop is
        a disc, and a few dozen copies of a bright drop are a swarm."""
        gl, ctx = self.gl, self.gl.ctx
        cu = dict(uC=cam["uC"], uF=cam["uF"], uR=cam["uR"], uU=cam["uU"], uFd=fd, uTan=TAN, uZoom=zoom, uAspect=self.aspect, uGc=tuple(float(v) for v in gc), uUniq=0)
        px_unit = self.size[1] / 2
        sunD = tuple(float(v) for v in -self.Lsun)
        res = tuple(float(x) for x in self.size)
        ctx.disable(moderngl.DEPTH_TEST)

        def samples(n, salt):
            pts = vogel(n, rot=(t * 3.1 + salt) % 6.283)
            rg = np.random.default_rng((int(t * 613) + salt) & 0xffff)
            pts = pts + (rg.random((n, 2)) - .5) * 1.4 * np.sqrt(np.pi / n) * .5
            rr = np.hypot(pts[:, 0], pts[:, 1])
            pts[rr > 1] /= rr[rr > 1, None]
            rim = .86 + .28 * (pts ** 2).sum(1)
            jit = (rg.random((n, 2)) - .5) * np.array([2 / self.size[0], 2 / self.size[1]])
            return pts, rim / rim.sum(), jit

        def blur(tex, zr, n):
            off = max(1 / zr[1] - fd, fd - 1 / zr[0], 0.0)
            sigma = .62 * K_COC * off * zoom * px_unit * np.sqrt(np.pi / n)
            if sigma > .35:
                gl.run("flower_gauss.frag", self.t_tmp, uTex=tex, uDir=(1 / self.size[0], 0.0), uSigma=float(sigma))
                gl.run("flower_gauss.frag", tex, uTex=self.t_tmp, uDir=(0.0, 1 / self.size[1]), uSigma=float(sigma))

        # the other flower's water: its own layer, behind the near flower, so nothing has to cut it
        self.t_beads2.clear()
        zr2 = self.second.zrange
        coc2 = K_COC * max(1 / zr2[1] - fd, fd - 1 / zr2[0], 0.0) * zoom * px_unit
        if coc2 < 170:      # past this a drop that size has spread out to nothing
            n2 = 96
            pts, rim, jit = samples(n2, 40)
            self.t_beads2.fbo.use()
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            for i in range(n2):
                pu = dict(cu, uLens=tuple(float(v) for v in pts[i] * A_LENS), uJit=tuple(float(v) for v in jit[i]))
                for k in range(4):
                    v = self.bead_vaos.get(("beads2", k))
                    if v is None:
                        continue
                    s = self.second.surf[k]
                    gl.bind(self.p_bead, dict(pu, uSway=sway_s, uPivot=self.second.pivot, uFlut=self._flut(t, k), uLit=s["lit"], uPig=s["pig"], uEnv=env, uSunD=sunD, uSunE=(.55, .42, .28),
                                              uGround=(.030, .011, .003), uRes=res, uKcoc=K_COC, uPxPerUnit=px_unit, uN=float(n2), uW=float(rim[i])))
                    v[0].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=v[1])
            ctx.disable(moderngl.BLEND)
            blur(self.t_beads2, zr2, n2)
        # this flower's
        self.t_beads.clear()
        self.t_beads.fbo.use()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        Nb = int(st.get("bead_passes", self.N_BEAD))
        pts, rim, jit = samples(Nb, 0)
        rq = self.runner(t)
        if rq is not None:
            inst = np.array([[*rq["P"], *rq["n"], *rq["t"], *rq["b"], rq["R"], rq["e"], 0, rq["zc"], rq["u"], rq["v"], rq["dudm"], rq["dvdm"], 0.0 if rq["free"] else rq["w"], 0]], "f4")
            self.run_buf.write(inst.tobytes())
        aq = self.arrival.at(t)
        if aq is not None:
            self.arr_buf.write(np.array([[*aq["P"], *aq["n"], *aq["t"], *aq["b"], aq["R"], aq["e"], 0, aq["zc"], aq["u"], aq["v"], aq["dudm"], aq["dvdm"], aq["w"], 0]], "f4").tobytes())
        n_trail = self.arrival.trail_count(t)
        alive = {key: self.arrival.alive(key, t, v[1]) for key, v in self.bead_vaos.items()}
        for i in range(Nb):
            pu = dict(cu, uLens=tuple(float(v) for v in pts[i] * A_LENS), uJit=tuple(float(v) for v in jit[i]))
            for k in range(4):
                s = self.hero.surf[k]
                common = dict(pu, uSway=sway_h, uPivot=self.hero.pivot, uFlut=self._flut(t, k), uLit=s["lit"], uPig=s["pig"], uEnv=env, uSunD=sunD, uSunE=(.55, .42, .28),
                              uGround=(.030, .011, .003), uRes=res, uKcoc=K_COC, uPxPerUnit=px_unit, uN=float(Nb), uW=float(rim[i]))
                for name in ("beads", "micro"):
                    v = self.bead_vaos.get((name, k))
                    if v is None:
                        continue
                    gl.bind(self.p_bead, common)
                    v[0].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=alive[(name, k)])
            if rq is not None:
                s = self.hero.surf[3]
                gl.bind(self.p_bead, dict(pu, uSway=sway_h, uPivot=self.hero.pivot, uFlut=0.0 if rq["free"] else self._flut(t, 3), uLit=s["lit"], uPig=s["pig"], uEnv=env, uSunD=sunD, uSunE=(.55, .42, .28),
                                          uGround=(.030, .011, .003), uRes=res, uKcoc=K_COC, uPxPerUnit=px_unit, uN=float(Nb), uW=float(rim[i])))
                self.v_run.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=1)
            if aq is not None or n_trail:
                ka = self.arrival.ip
                s = self.hero.surf[ka]
                gl.bind(self.p_bead, dict(pu, uSway=sway_h, uPivot=self.hero.pivot, uFlut=self._flut(t, ka), uLit=s["lit"], uPig=s["pig"], uEnv=env, uSunD=sunD, uSunE=(.55, .42, .28),
                                          uGround=(.030, .011, .003), uRes=res, uKcoc=K_COC, uPxPerUnit=px_unit, uN=float(Nb), uW=float(rim[i])))
                if aq is not None:
                    self.v_arr.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=1)
                if n_trail:
                    self.v_trail.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n_trail)
            if i % 16 == 15:
                ctx.finish()
        ctx.disable(moderngl.BLEND)
        # a drop nowhere near focus: its copies are far apart however many there are
        blur(self.t_beads, self.hero.zrange, Nb)

    # ---- the one that lets go ----------------------------------------------------------------
    def _runner_setup(self):
        pt = self.data["hero"][3]
        self.run_pt = pt
        N = pt["mesh"]["P"].shape[0]
        self.run_path = geo.downhill(pt, (N * .50, N * .46))
        print(f"[flower] runner path {len(self.run_path)} steps", flush=True)
        self.run_buf = self.gl.ctx.buffer(reserve=24 * 4, dynamic=True)
        self.v_run = self.gl.instanced(self.p_bead, "3f 3f 3f 3f 4f 4f 2f/i", ["ic", "inb", "it", "ib", "ishape", "iuv", "iw"], self.run_buf)

    def _run_s(self, t):
        """it clings, it creeps, and then it goes."""
        t0, t1 = self.T_RUN
        u = np.clip((t - t0 - .75) / (t1 - t0 - .75), 0, 1)
        return float(u ** 2.6)

    def runner(self, t):
        """-> dict(P, n, t, b, R, uv, ...) or None."""
        t0, t1 = self.T_RUN
        if t < t0:
            return None
        m = self.run_pt["mesh"]
        path = self.run_path
        L, Phi = self.run_pt["meta"]["L"], self.run_pt["meta"]["Phi"]
        N = m["P"].shape[0]
        if t <= t1:
            k = self._run_s(t) * (len(path) - 1)
            i, j = path[int(k)] + (path[min(int(k) + 1, len(path) - 1)] - path[int(k)]) * (k - int(k))
            i0, j0 = int(np.clip(i, 0, N - 2)), int(np.clip(j, 0, N - 2))
            fi, fj = i - i0, j - j0
            ip = lambda a: (a[i0, j0] * (1 - fi) * (1 - fj) + a[i0 + 1, j0] * fi * (1 - fj) + a[i0, j0 + 1] * (1 - fi) * fj + a[i0 + 1, j0 + 1] * fi * fj)
            P, n, tt, b = ip(m["P"]), ip(m["n"]), ip(m["t"]), ip(m["b"])
            # the side of the sheet that faces up
            if n[1] < 0:
                n = -n
            r = (i + .5) / N * L
            u, v = (j + .5) / N, (i + .5) / N
            # it lands: for a moment it rings
            w = np.exp(-7 * (t - t0)) * np.sin(38 * (t - t0))
            speed = (self._run_s(t + .02) - self._run_s(t - .02)) * len(path) / .04
            e = 1.0 + min(speed * .002, .45)
            return dict(P=P, n=n, t=tt, b=b, R=.00185 * (1 + .22 * w) * min((t - t0) / .06 + .3, 1.0), e=e, u=u, v=v, dudm=1 / (2 * Phi * max(r, .002)), dvdm=1 / L, w=(i / N) ** 2, zc=.30, free=False)
        # off the end of the petal, and gone
        q = self.runner(t1 - 1e-4)
        dt = t - t1
        if dt > .4:
            return None
        P = q["P"] + np.array([0, -.5 * 9.8 * dt * dt - .12 * dt, -.05 * dt], "f4")
        return dict(q, P=P, n=np.array([0, 1, 0], "f4"), t=np.array([1, 0, 0], "f4"), b=np.array([0, 0, 1], "f4"), e=1.0, zc=1.05, free=True)

    def runner_view(self, t):
        q = self.runner(t)
        if q is None:
            return None
        P = q["P"]
        return float(P[0] / P[2] / TAN), float(P[1] / P[2] / TAN), float(P[2])

    def _lens(self, G, sway, t, fd, zoom, gc, cam, env, N, beads, acc, tmp, seed, sd):
        gl, ctx = self.gl, self.gl.ctx
        pts = vogel(N, rot=(t * 7.3 + seed) % 6.283)
        # jittered, or the copies of an edge line up into lines of their own
        rg = np.random.default_rng((int(t * 977) + sd * 13) & 0xffff)
        pts = pts + (rg.random((N, 2)) - .5) * 1.4 * np.sqrt(np.pi / N) * .5
        rr = np.hypot(pts[:, 0], pts[:, 1])
        pts[rr > 1] /= rr[rr > 1, None]
        rim = .86 + .28 * (pts ** 2).sum(1)
        sunD = tuple(float(v) for v in -self.Lsun)
        rng = np.random.default_rng((int(t * 1000) + sd) & 0xffff)
        jit = (rng.random((N, 2)) - .5) * np.array([2 / self.size[0], 2 / self.size[1]])
        px_unit = self.size[1] / 2
        blur_k = 1.15 * np.sqrt(np.pi / N)
        acc.clear()
        for i in range(N):
            a = tuple(float(v) for v in pts[i] * A_LENS)
            pass_u = dict(uC=cam["uC"], uF=cam["uF"], uR=cam["uR"], uU=cam["uU"], uLens=a, uFd=fd, uTan=TAN, uZoom=zoom, uAspect=self.aspect,
                          uGc=tuple(float(v) for v in gc), uJit=tuple(float(v) for v in jit[i]), uSpace=0)
            self.fb_pass.use()
            self.fb_pass.clear(0, 0, 0, 0, depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST)
            for k, s in enumerate(G.surf):
                gl.bind(G.p_mesh, dict(pass_u, uSway=sway, uPivot=G.pivot, uFlut=self._flut(t, k) if k < 4 else 0.0, uPig=s["pig"], uDet=s["det"], uLit=s["lit"], uEnv=env,
                                       uSunD=sunD, uSunE=(1.6, 1.25, .9), uKcoc=K_COC, uPxPerUnit=px_unit, uBlurK=blur_k, uKDet=.8, uGloss=.11 if k < 4 else .04))
                s["a"].render(moderngl.TRIANGLES)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            gl.run("flower_add.frag", acc, uTex=self.t_pass, uW=float(rim[i]) / float(rim.sum()))
            ctx.disable(moderngl.BLEND)
            if i % 8 == 7:
                ctx.finish()
        # however far the nearest part of this flower is from the focal plane, its copies are that far apart
        zr = G.zrange
        d0, d1 = 1 / zr[1], 1 / zr[0]
        off = max(d0 - fd, fd - d1, 0.0)
        coc = K_COC * off * zoom * px_unit
        sigma = .62 * coc * np.sqrt(np.pi / N)
        if sigma > .35:
            gl.run("flower_gauss.frag", tmp, uTex=acc, uDir=(1 / self.size[0], 0.0), uSigma=float(sigma))
            gl.run("flower_gauss.frag", acc, uTex=tmp, uDir=(0.0, 1 / self.size[1]), uSigma=float(sigma))
