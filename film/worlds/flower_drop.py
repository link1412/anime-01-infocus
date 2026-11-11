"""One drop, hanging from the low point of a petal's edge.

Its shape is what water does: the Young-Laplace equation, its weight against its skin, integrated from the
bottom up. The light in it is followed, surface by surface: what comes off the front, what goes through and
out of the back, what bounces inside first, what lands on the underside of the petal it hangs from. What it
sees out there is the rest of this scene, drawn again from where the drop is, and the sea, drawn again for it.
Nothing about the picture inside it is decided here; it is only where those rays end up.
"""
from __future__ import annotations
import numpy as np
import moderngl
from scipy.spatial import cKDTree
from ..core import smooth

RHO_G_OVER_GAMMA = 998 * 9.81 / .0728       # 1/m^2, water
APEX_B = .0017          # radius of curvature at its lowest point. a drop with nearly as much water as a petal can hold
R_EQ = .00225           # metres, how wide it is at its widest. a little more than water that shape would be: nobody has measured it
MARGIN_Y = 1.73         # how far above the widest point the edge of the petal is, in R
TAPER = .86             # less water than a drop that is about to fall: the same curve, a shorter neck
MENISCUS = (.20, .075)  # how much further out, and over what height, the water reaches where it meets the sheet
INSET = .58             # how far back under the sheet the axis stands, in R: it hangs from what it wets
ETA_REF = 1.333
FAR_YAW, FAR_PITCH = -.172, -.10    # the sea, as seen from the flower: the same numbers it is seen behind the flower with
TAN_WIDE = 1.0
CUBE = 1024
GUARD = 16
BEHIND, TAN_BEHIND = 2048, .6
PROF = (512, 1024)
ROOM = 1.95             # half size of the box the table covers, in R
Y_BOUND = .52           # centre of the bounding sphere above the widest point
SHEET = 1024
SHEET_L = 2.0
# faces of the box around the drop: forward, right, up, where in the atlas
FACES = (((1, 0, 0), (0, 0, -1), (0, 1, 0), (0, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0)),
         ((0, 1, 0), (1, 0, 0), (0, 0, -1), (2, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1), (0, 1)),
         ((0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0), (2, 1)))


def young_laplace(b=APEX_B, flare=.80):
    """the meridian of a pendant drop: r(s), y(s) from the apex, scaled to its widest radius, y=0 at the widest point."""
    ds = b / 2000
    x, z, phi = 1e-7, 0.0, 0.0
    pts = [(x, z)]

    def f(x, z, phi):
        k = 2 / b - RHO_G_OVER_GAMMA * z - (np.sin(phi) / x if x > 1e-6 else 1 / b)
        return np.cos(phi), np.sin(phi), k
    stage = 0
    eq = None
    for i in range(40000):
        a = f(x, z, phi)
        c = f(x + ds / 2 * a[0], z + ds / 2 * a[1], phi + ds / 2 * a[2])
        d = f(x + ds / 2 * c[0], z + ds / 2 * c[1], phi + ds / 2 * c[2])
        e = f(x + ds * d[0], z + ds * d[1], phi + ds * d[2])
        x += ds / 6 * (a[0] + 2 * c[0] + 2 * d[0] + e[0])
        z += ds / 6 * (a[1] + 2 * c[1] + 2 * d[1] + e[1])
        phi += ds / 6 * (a[2] + 2 * c[2] + 2 * d[2] + e[2])
        pts.append((x, z))
        if stage == 0 and np.cos(phi) < 0:
            stage, eq = 1, (x, z)
        elif stage == 1 and np.cos(phi) > 0:
            stage = 2
        elif stage == 2 and x / eq[0] > flare:
            break
    p = np.array(pts)
    r, y = p[:, 0] / eq[0], (p[:, 1] - eq[1]) / eq[0]
    return r, y


def closed_meridian():
    """up the side, then shut. what shuts it is above the sheet, where nothing looks."""
    r, y = young_laplace()
    y = np.where(y > 0, y * TAPER, y)
    top = MARGIN_Y + .10
    keep = y <= top
    r, y = r[keep], y[keep]
    r = r + MENISCUS[0] * np.exp((y - top) / MENISCUS[1])
    rc = .05
    a = np.linspace(0, np.pi / 2, 40)[1:]
    r = np.concatenate([r, r[-1] - rc + rc * np.cos(a), np.linspace(r[-1] - rc, 0, 40)[1:]])
    y = np.concatenate([y, y[-1] + rc * np.sin(a), np.full(39, y[-1] + rc)])
    # even steps along it
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(r), np.diff(y)))])
    u = np.linspace(0, s[-1], 24000)
    return np.interp(u, s, r), np.interp(u, s, y)


def profile_table():
    """signed distance to the meridian and the normal of the nearest point of it, over (r, y)."""
    r, y = closed_meridian()
    # outward normal from the tangent
    k = np.ones(25) / 25
    dr, dy = np.gradient(r), np.gradient(y)
    dr[12:-12], dy[12:-12] = np.convolve(dr, k, "valid"), np.convolve(dy, k, "valid")
    ln = np.maximum(np.hypot(dr, dy), 1e-9)
    nr, ny = dy / ln, -dr / ln
    W, H = PROF
    y0, span = Y_BOUND - ROOM, 2 * ROOM
    gr = (np.arange(W) + .5) / W * ROOM
    gy = y0 + (np.arange(H) + .5) / H * span
    G = np.stack(np.meshgrid(gr, gy), -1).reshape(-1, 2)
    dist, idx = cKDTree(np.stack([r, y], 1)).query(G)
    order = np.argsort(y)
    rofy = np.interp(G[:, 1], y[order], r[order])
    inside = (G[:, 1] > y.min()) & (G[:, 1] < y.max()) & (G[:, 0] < rofy)
    sd = np.where(inside, -dist, dist)
    a = np.stack([sd, nr[idx], ny[idx], np.ones_like(sd)], 1).astype("f4").reshape(H, W, 4)
    return a, (ROOM, y0, span), float(y.max()), float(y.min())


def find_site(hero, target, tan):
    """the low point of an edge: where water that runs along it stops."""
    best, bd = None, 1e9
    for ip in range(4):
        s = hero[ip]
        P = s["mesh"]["P"]
        N = P.shape[0]
        pig = s["pig"]
        T = pig.shape[0]
        row = []
        for j in range(4, N - 4):
            nz = np.nonzero(pig[:, int((j + .5) / N * T), 3] > .5)[0]
            row.append(min(int((nz.max() + .5) / T * N), N - 1) if len(nz) else -1)
        row = np.array(row)
        for c in range(12, len(row) - 12):
            i, j = row[c], c + 4
            if i < N * .5:
                continue
            ys = np.array([P[row[c + o], c + o + 4, 1] for o in range(-12, 13) if row[c + o] >= 0])
            if P[i, j, 1] > ys.min() + 1e-6 or ys.max() - P[i, j, 1] < .00018:
                continue
            q = P[i, j]
            v = np.array([q[0] / q[2] / tan, q[1] / q[2] / tan])
            d = float(np.linalg.norm(v - target))
            if d < bd:
                bd, best = d, (ip, int(i), int(j))
    ip, i, j = best
    m = hero[ip]["mesh"]
    P = m["P"]
    q = P[i, j].astype("f8")
    N = P.shape[0]
    pu = (P[i, min(j + 4, N - 1)] - P[i, max(j - 4, 0)]).astype("f8") / (8 / N)
    pv = (P[i, j] - P[max(i - 8, 0), j]).astype("f8") / (8 / N)
    t = pu / np.linalg.norm(pu)
    inward = -(pv / np.linalg.norm(pv))
    inward = inward - t * (inward @ t)
    inward[1] = 0.0
    inward /= np.linalg.norm(inward)
    n = m["n"][i, j].astype("f8")
    n = n if n[1] > 0 else -n
    return dict(ip=ip, i=i, j=j, q=q, n=n, inward=inward, w=float(m["w"][i, j]) if m["w"].ndim == 2 else (i / N) ** 2)


class HangingDrop:
    SAMPLES = ((3.0, 320), (8.0, 192), (16.0, 128), (30.0, 96), (1e9, 64))

    def __init__(self, fl, tan, k_coc, a_lens, target=(.533, .015)):
        self.fl, self.gl = fl, fl.gl
        self.tan, self.k_coc, self.a_lens = tan, k_coc, a_lens
        gl, ctx = fl.gl, fl.gl.ctx
        self.size = fl.size
        self.site = s = find_site(fl.data["hero"], np.array(target), tan)
        self.R = R_EQ
        self.q0 = s["q"]
        self.c0 = s["q"] + s["inward"] * INSET * R_EQ + np.array([0, -MARGIN_Y * R_EQ, 0])
        z = float(self.c0[2])
        self.z = z
        self.rv = R_EQ / (z * tan)
        self.view = (float(self.c0[0] / z / tan), float(self.c0[1] / z / tan))
        table, self.prof_win, self.y_top, self.y_apex = profile_table()
        self.t_prof = ctx.texture((PROF[0], PROF[1]), 4, table.tobytes(), dtype="f4")
        self.t_prof.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_prof.repeat_x = self.t_prof.repeat_y = False
        self.view_apex = (self.view[0], float((self.c0[1] + self.y_apex * R_EQ) / z / tan))
        self.view_q = (float(self.q0[0] / self.q0[2] / tan), float(self.q0[1] / self.q0[2] / tan))
        # where its image of the far side stands: in front of it, by about the drop's own radius
        self.f_surface = 1 / (z - R_EQ)
        self.f_image = 1 / (z - 2.015 * R_EQ)
        print(f"[flower] the drop hangs from petal {s['ip']} (j {s['j']}), view ({self.view[0]:.4f},{self.view[1]:.4f}), edge ({self.view_q[0]:.4f},{self.view_q[1]:.4f}), "
              f"apex y {self.view_apex[1]:.4f}, R {self.rv:.4f} view ({R_EQ * 1000:.2f} mm), z {z:.4f}, in focus {self.f_surface:.3f} D, its picture {self.f_image:.3f} D", flush=True)
        self._bake_sheet()
        # the world from here
        self.p_env = gl.prog("flower_pass.frag", "flower_env.vert")
        fmt, names = "3f 3f 3f 3f 2f 1f", ["in_p", "in_n", "in_t", "in_b", "in_uv", "in_w"]
        self.vaos = {}
        for G in (fl.hero, fl.second):
            for k, srf in enumerate(G.surf):
                self.vaos[(G.name, k)] = ctx.vertex_array(self.p_env, [(srf["vb"], fmt, *names)], srf["ib"], skip_errors=True)
        self.atlas = gl.target((3 * CUBE, 2 * CUBE), mips=True)
        self.d_atlas = ctx.depth_renderbuffer(self.atlas.size)
        self.fb_atlas = ctx.framebuffer([self.atlas.tex], self.d_atlas)
        self.tan_cube = 1 / (1 - 2 * GUARD / CUBE)
        self.behind = gl.target((BEHIND, BEHIND), mips=True)
        self.d_behind = ctx.depth_renderbuffer((BEHIND, BEHIND))
        self.fb_behind = ctx.framebuffer([self.behind.tex], self.d_behind)
        self.t_wide = gl.target(self.size, mips=True)
        self.t_near = gl.target(self.size, mips=True)
        self.t_acc = gl.target(self.size, dtype="f4")
        self.b0 = self._basis(FAR_YAW, FAR_PITCH)

    @staticmethod
    def _basis(yaw, pitch):
        cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        F = np.array([sy * cp, sp, cy * cp])
        R = np.array([cy, 0.0, -sy])
        return F, R, np.cross(F, R)

    def _bake_sheet(self):
        """the petal as seen from straight above the drop: its height and its place on the sheet, for rays inside the water."""
        gl, ctx = self.gl, self.gl.ctx
        srf = self.fl.hero.surf[self.site["ip"]]
        y0, y1 = MARGIN_Y - .55, MARGIN_Y + .85
        tg = ctx.texture((SHEET, SHEET), 4, dtype="f4")
        tg.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tg.repeat_x = tg.repeat_y = False
        dp = ctx.depth_renderbuffer((SHEET, SHEET))
        fb = ctx.framebuffer([tg], dp)
        prog = gl.prog("flower_sheet.frag", "flower_sheet.vert")
        vao = ctx.vertex_array(prog, [(srf["vb"], "3f 3f 3f 3f 2f 1f", "in_p", "in_n", "in_t", "in_b", "in_uv", "in_w")], srf["ib"], skip_errors=True)
        fb.use()
        fb.clear(0, 0, 0, 0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        gl.bind(prog, dict(uO=tuple(float(v) for v in self.c0), uRd=self.R, uL=SHEET_L, uY0=y0, uY1=y1))
        vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.DEPTH_TEST)
        a = np.frombuffer(fb.read(components=4, dtype="f4"), "f4").reshape(SHEET, SHEET, 4)
        h = a[..., 0][a[..., 3] > .5]
        self.sheet_y = (float(h.min()) - .01, float(h.max()) + .01) if h.size else (9.0, 9.0)
        self.t_sheet = tg
        print(f"[flower] sheet over the drop: y {self.sheet_y[0]:.3f} to {self.sheet_y[1]:.3f} R, {100 * h.size / SHEET ** 2:.0f}% of the map", flush=True)

    # ---- how it moves --------------------------------------------------------------------------
    def _motion(self, t):
        """rain lands on the flower; the drop has the slowest modes of anything here, and is last to stop."""
        fl = self.fl
        st, tx, tz = 0.0, 0.0, 0.0
        for t0, k, a in fl.IMPACTS:
            if t <= t0:
                continue
            u = t - t0
            A = abs(a) / .001 * (1.0 if k == self.site["ip"] else .22)
            st += .030 * A * np.exp(-3.0 * u) * np.sin(2 * np.pi * 10.5 * u)
            sw = .045 * A * np.exp(-1.7 * u) * np.sin(2 * np.pi * 4.3 * u + k)
            tx += sw * .6 * np.sign(a)
            tz += sw * np.sign(a)
        st += .0018 * np.sin(t * 8.1) + .0011 * np.sin(t * 19.3 + 1.0)
        # and it has only just got here: it comes in along the sheet, down and to the left, toward the camera
        a_st, a_sw = fl.arrival.ring(t)
        st += a_st
        tx += a_sw * .45
        tz -= a_sw
        return 1 + st, tx, tz

    def frame(self, t):
        fl = self.fl
        G = fl.hero
        ip = self.site["ip"]
        S = np.array(fl._sway(t, "hero"), "f8").reshape(3, 3).T
        p = np.array(G.pivot, "f8")
        q = p + S @ (self.q0 + self.site["n"] * self.site["w"] * fl._flut(t, ip) - p)
        e, tx, tz = self._motion(t)
        cx, sx, cz, sz = np.cos(tx), np.sin(tx), np.cos(tz), np.sin(tz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Rt = Rz @ Rx
        # it hangs from the edge, however much of it has got here
        c = q + Rt @ (self.c0 - self.q0) * fl.arrival.swell(t)
        return c, Rt, e, q

    # ---- what it sees ---------------------------------------------------------------------------
    def _capture(self, t, C, sway_h, sway_s, env):
        fl = self.fl
        gl, ctx = self.gl, self.gl.ctx
        sunD = tuple(float(v) for v in -fl.Lsun)
        base = dict(uEye=tuple(float(v) for v in C), uC=tuple(float(v) for v in C), uEnv=env, uSunD=sunD, uSunE=(1.6, 1.25, .9), uKcoc=0.0, uFd=0.0, uZoom=1.0, uPxPerUnit=1.0, uBlurK=1.0, uKDet=.8)
        ctx.enable(moderngl.DEPTH_TEST)

        def draw(F, R, U, tan):
            for G, sway in ((fl.hero, sway_h), (fl.second, sway_s)):
                for k, srf in enumerate(G.surf):
                    gl.bind(self.p_env, dict(base, uFaceF=F, uFaceR=R, uFaceU=U, uTanE=float(tan), uSway=sway, uPivot=G.pivot, uFlut=fl._flut(t, k) if k < 4 else 0.0,
                                             uPig=srf["pig"], uDet=srf["det"], uLit=srf["lit"], uGloss=.11 if k < 4 else .04))
                    self.vaos[(G.name, k)].render(moderngl.TRIANGLES)
        fb = self.fb_atlas
        fb.use()
        fb.scissor = None
        fb.viewport = (0, 0, *fb.size)
        fb.clear(0, 0, 0, 0, depth=1.0)
        for F, R, U, (tx, ty) in FACES:
            fb.viewport = (tx * CUBE, ty * CUBE, CUBE, CUBE)
            draw(tuple(float(v) for v in F), tuple(float(v) for v in R), tuple(float(v) for v in U), self.tan_cube)
        fb.viewport = (0, 0, *fb.size)
        self.atlas.gen()
        fb = self.fb_behind
        fb.use()
        fb.clear(0, 0, 0, 0, depth=1.0)
        draw((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), TAN_BEHIND)
        self.behind.gen()
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.finish()

    def rho_dir(self, C, rho):
        """where a ray that lands on a ball at rho (in radii, as the camera sees it) ends up, in the flower's frame."""
        look = C / np.linalg.norm(C)
        rx = np.array([1.0, 0, 0]) - look * look[0]
        rx /= np.linalg.norm(rx)
        ry = np.cross(look, rx)
        h = min(float(np.hypot(*rho)), .8)
        if h < 1e-6:
            return look
        a = 2 * (np.arcsin(h) - np.arcsin(h / ETA_REF))
        e = (rx * rho[0] + ry * rho[1]) / max(float(np.hypot(*rho)), 1e-9)
        return look * np.cos(a) - e * np.sin(a)

    def morph(self, zoom):
        return float(smooth(np.log(26.0), np.log(46.2), np.log(zoom)))

    def far_gain(self, zoom):
        """behind the flower the beach is kept low key. as the drop fills the frame the camera comes round to the drop's own exposure."""
        fk = self.fl._far_k if self.fl._far_k is not None else 1.0
        k = float(smooth(np.log(3.0), np.log(46.8), np.log(max(zoom, 1.0))))
        return float(np.exp(np.log(fk) * (1 - k))), k

    def _far_field(self, st, t, zoom, gc, C, k):
        sea = self.fl.sea
        end = st.get("end", (0.0, -.095, .27))
        if sea is None:
            if not hasattr(self, "_stubbed"):
                self._stubbed = True
                w, h = self.size
                yy, xx = np.mgrid[0:h, 0:w].astype("f4")
                ndc_x, ndc_y = (xx + .5) / w * 2 - 1, (yy + .5) / h * 2 - 1
                sky = ndc_y * TAN_WIDE > .1
                a = np.where(sky[..., None], np.array([.8, .7, .6], "f4") * (1 + .5 * np.sin(ndc_x * 40) * np.sin(ndc_y * 40))[..., None], np.array([.05, .05, .06], "f4"))
                self.t_wide.tex.write(np.concatenate([a, np.ones((h, w, 1), "f4")], -1).astype("f2").tobytes())
                self.t_wide.gen()
            return dict(F=self.b0[0], R=self.b0[1], U=self.b0[2], tan=TAN_WIDE, has=0.0, end=end)
        sea.render(dict(t=t, yaw=FAR_YAW, pitch=FAR_PITCH, roll=0.0, tan_half=TAN_WIDE, yaw_v=0.0, drop=None, crack=None), self.t_wide)
        self.t_wide.gen()
        hh = 1 / (zoom * self.rv)
        out = dict(F=self.b0[0], R=self.b0[1], U=self.b0[2], tan=TAN_WIDE, has=0.0, end=end)
        if zoom > 3.0:
            rho = ((gc[0] - self.view[0]) / self.rv, (gc[1] - self.view[1]) / self.rv)
            d = self.rho_dir(C, rho)
            ds = d[0] * self.b0[1] + d[1] * self.b0[2] + d[2] * self.b0[0]
            yaw_c, pitch_c = float(np.arctan2(ds[0], ds[2])), float(np.arcsin(np.clip(ds[1], -1, 1)))
            tan_c = float(np.clip(.6 * hh, .27, .55))
            yaw = yaw_c + (end[0] - yaw_c) * k
            pitch = pitch_c + (end[1] - pitch_c) * k
            tan = tan_c + (end[2] - tan_c) * k
            sea.render(dict(t=t, yaw=yaw, pitch=pitch, roll=0.0, tan_half=tan, yaw_v=0.0, drop=None, crack=None), self.t_near)
            self.t_near.gen()
            F, R, U = self._basis(yaw, pitch)
            out.update(has=1.0, Fh=F, Rh=R, Uh=U, tanh=tan)
        return out

    def probe(self, view):
        """where the ray through this point of the frame ends up, as the sea's own yaw and pitch."""
        gl = self.gl
        tg = gl.target((8, 8), dtype="f4", nearest=True)
        srf = self.fl.hero.surf[self.site["ip"]]
        gl.run("flower_drop.frag", tg, uProbe=1.0, uC=(0.0, 0.0, 0.0), uF=(0.0, 0.0, 1.0), uR=(1.0, 0.0, 0.0), uU=(0.0, 1.0, 0.0), uFd=self.f_image, uTan=self.tan, uZoom=1e4, uAspect=1.0, uGc=tuple(float(v) for v in view),
               uRes=(8.0, 8.0), uDc=tuple(float(v) for v in self.c0), uDx=(1.0, 0.0, 0.0), uDy=(0.0, 1.0, 0.0), uDz=(0.0, 0.0, 1.0), uDr=self.R, uRv=self.rv, uPxV=1e4, uRhoS=0.0, uSt=(1.0, 1.0), uYp=MARGIN_Y,
               uProfW=tuple(float(v) for v in self.prof_win), uBound=(Y_BOUND, ROOM * .97), uSheetW=(SHEET_L, self.sheet_y[0], self.sheet_y[1]), uSheet=self.t_sheet, uProf=self.t_prof, uPigS=srf["pig"], uLitS=srf["lit"],
               uS0x=tuple(float(v) for v in self.b0[1]), uS0y=tuple(float(v) for v in self.b0[2]), uS0z=tuple(float(v) for v in self.b0[0]), uLens=(0.0, 0.0), uJit=(0.0, 0.0), uW=1.0, uMorph=0.0, uMapS=1.0, uMapO=(0.0, 0.0, 0.0))
        a = np.frombuffer(tg.fbo.read(components=4, dtype="f4"), "f4").reshape(8, 8, 4)[4, 4]
        gl.bind(gl.prog("flower_drop.frag"), dict(uProbe=0.0))
        return float(np.arctan2(a[0], a[2])), float(np.arcsin(np.clip(a[1], -1, 1)))

    # ---- a frame -------------------------------------------------------------------------------
    def render(self, st, t, fd, zoom, gc, cam, env, sway_h, sway_s):
        fl = self.fl
        gl, ctx = self.gl, self.gl.ctx
        res = self.size
        swell = fl.arrival.swell(t)
        if swell <= 0:      # nothing has got here yet
            self.t_acc.clear()
            return
        R, rv = self.R * swell, self.rv * swell
        C, Rt, e, q = self.frame(t)
        k = st.get("morph", self.morph(zoom))
        self._capture(t, C, sway_h, sway_s, env)
        far = self._far_field(st, t, zoom, gc, C, k)
        fk, _ = self.far_gain(zoom)
        n = int(st.get("drop_passes", 0)) or next(v for z, v in self.SAMPLES if zoom < z)
        stop = float(st.get("stop", 1.0))
        # lens pattern
        i = np.arange(n) + .5
        rot = (t * 3.1) % 6.283
        lens = np.stack([np.sqrt(i / n) * np.cos(i * 2.39996323 + rot), np.sqrt(i / n) * np.sin(i * 2.39996323 + rot)], 1)
        rim = .86 + .28 * (lens ** 2).sum(1)
        rim /= rim.sum()
        rg = np.random.default_rng((int(t * 613) + 91) & 0xffff)
        jit = (rg.random((n, 2)) - .5) * np.array([2 / res[0], 2 / res[1]])
        # how much of the frame this can possibly touch
        z = float(C[2])
        vc = (float(C[0] / z / self.tan), float(C[1] / z / self.tan))
        Ry = (Y_BOUND * R) / (z * self.tan)
        coc = self.k_coc * stop * (abs(fd - 1 / z) + .06)
        hw = ROOM * rv + coc + 2 / (zoom * res[1] / 2)
        px = np.array(res, "f8")
        c0 = ((vc[0] - gc[0]) * zoom / fl.aspect * .5 + .5) * px[0], ((vc[1] + Ry - gc[1]) * zoom * .5 + .5) * px[1]
        rp = hw * zoom * res[1] / 2
        x0, x1 = int(np.floor(max(c0[0] - rp, 0))), int(np.ceil(min(c0[0] + rp, px[0])))
        y0, y1 = int(np.floor(max(c0[1] - rp, 0))), int(np.ceil(min(c0[1] + rp, px[1])))
        self.t_acc.clear()
        if x1 <= x0 or y1 <= y0:
            return
        px_v = zoom * res[1] / 2
        u = dict(
            uC=cam["uC"], uF=cam["uF"], uR=cam["uR"], uU=cam["uU"], uFd=fd, uTan=self.tan, uZoom=zoom, uAspect=fl.aspect, uGc=tuple(float(v) for v in gc), uRes=tuple(float(v) for v in res),
            uDc=tuple(float(v) for v in C), uDx=tuple(float(v) for v in Rt[:, 0]), uDy=tuple(float(v) for v in Rt[:, 1]), uDz=tuple(float(v) for v in Rt[:, 2]),
            uDr=R, uRv=rv, uPxV=float(px_v), uRhoS=float(self.k_coc * stop * abs(fd - self.f_image) / rv * np.sqrt(np.pi / n)),
            uMapS=float(swell), uMapO=tuple(float(v) for v in (self.q0 - self.c0) / self.R * (1 - swell)),
            uSt=(float(1 / np.sqrt(e)), float(e)), uYp=MARGIN_Y, uProfW=tuple(float(v) for v in self.prof_win), uBound=(Y_BOUND, ROOM * .97),
            uSheetW=(SHEET_L, self.sheet_y[0], self.sheet_y[1]), uAtlas=self.atlas, uBehind=self.behind, uFarW=self.t_wide, uFarH=self.t_near, uEnv=env,
            uS0x=tuple(float(v) for v in self.b0[1]), uS0y=tuple(float(v) for v in self.b0[2]), uS0z=tuple(float(v) for v in self.b0[0]),
            uFw=tuple(float(v) for v in self.b0[0]), uRw=tuple(float(v) for v in self.b0[1]), uUw=tuple(float(v) for v in self.b0[2]), uTanW=TAN_WIDE, uHasH=far["has"],
            uTanC=float(self.tan_cube), uTanB=TAN_BEHIND, uPxC=CUBE / (2 * self.tan_cube), uPxB=BEHIND / (2 * TAN_BEHIND), uAspS=fl.aspect, uFarK=fk, uFarMax=float(12 - 6 * smooth(0.0, .15, k)), uMorph=k,
        )
        if far["has"] > 0:
            u.update(uFh=tuple(float(v) for v in far["Fh"]), uRh=tuple(float(v) for v in far["Rh"]), uUh=tuple(float(v) for v in far["Uh"]), uTanH=far["tanh"])
        Fe, Re, Ue = self._basis(far["end"][0], far["end"][1])
        u.update(uFe=tuple(float(v) for v in Fe), uRe=tuple(float(v) for v in Re), uUe=tuple(float(v) for v in Ue), uTanE=float(far["end"][2]))
        srf = fl.hero.surf[self.site["ip"]]
        u.update(uSheet=self.t_sheet, uProf=self.t_prof, uLitS=srf["lit"], uPigS=srf["pig"], uLodS=float(max(np.log2(max(self.R / (self.rv * px_v) * fl.data["hero"][self.site["ip"]]["meta"]["L"] ** -1 * srf["lit"].size[0], 1.0)) + .5, 0.0)))
        prog = gl.prog("flower_drop.frag")
        self.t_acc.fbo.scissor = (x0, y0, x1 - x0, y1 - y0)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(n):
            ls = lens[i] * self.a_lens * stop
            if i == 0:
                gl.run("flower_drop.frag", self.t_acc, **dict(u, uLens=tuple(float(v) for v in ls), uJit=tuple(float(v) for v in jit[i]), uW=float(rim[i])))
                vao = gl._vaos[id(prog)]
            else:
                prog["uLens"].value = tuple(float(v) for v in ls)
                prog["uJit"].value = tuple(float(v) for v in jit[i])
                prog["uW"].value = float(rim[i])
                vao.render(moderngl.TRIANGLES, vertices=3)
            if i % 8 == 7:
                ctx.finish()
        ctx.disable(moderngl.BLEND)
        self.t_acc.fbo.scissor = None
