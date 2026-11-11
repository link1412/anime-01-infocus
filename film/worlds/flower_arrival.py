"""The second drop. The first one let go of its petal and the camera lost it. This one gets there.

It lands high on the sheet, rings, clings, and lets go. What pulls it is how much water it holds; what holds it
back is the width of what touches the sheet, so it goes faster for everything it swallows on the way, and it does
swallow them: the beads on its line are gone once it has been through, and what it leaves behind it is a line of
very small ones. At the low point of the edge it goes over. What hangs there is the water that came down, and then
what is still coming down behind it, so it swells, and swings, and only then is it still.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from ..core import smooth

T_LAND = 8.5        # when it lands
T_OVER = 10.2       # when its front gets to the lip
R_LAND = .00155     # metres
G = 9.8
PIN = 3.5           # m/s^2 of hold, for a drop of R_LAND, on a clean bit of sheet
SLOPE_MIN = .70     # the water behind it pushes as well: on the flat by the edge it still comes on
FILM = 1.1e-7       # what it picks up per metre it travels (R^3 / m), off a sheet that has been rained on for a while
SHARE = .50         # how much of what hangs there is the drop itself. the rest drains in behind it
TILT = .0016        # a drop this wide can't follow the sheet any closer than this (metres): it sits on the average of what it covers
# through the valleys, where the beads already are: (row, column) on the sheet. the last one is the low point of the edge
WAY = ((318, 106), (316, 126), (319, 140), (327, 150), (348, 160), (368, 170))


def _bilerp(a, ij):
    N = a.shape[0]
    i, j = ij[:, 0], ij[:, 1]
    i0, j0 = np.clip(i.astype(int), 0, N - 2), np.clip(j.astype(int), 0, a.shape[1] - 2)
    fi, fj = i - i0, j - j0
    if a.ndim == 3:
        fi, fj = fi[:, None], fj[:, None]
    return a[i0, j0] * (1 - fi) * (1 - fj) + a[i0 + 1, j0] * fi * (1 - fj) + a[i0, j0 + 1] * (1 - fi) * fj + a[i0 + 1, j0 + 1] * fi * fj


class Arrival:
    DT = 1 / 600

    def __init__(self, fl):
        self.fl = fl
        site = fl.drop.site
        self.ip = site["ip"]
        self.pt = fl.data["hero"][self.ip]
        self._path(site)
        self._simulate()
        self._swallowed()
        self._trail()
        print(f"[flower] the second drop: {self.L * 1000:.1f} mm of sheet, lets go at {self.t_go:.2f}, at the lip at {self.t_front:.2f}, "
              f"R {R_LAND * 1000:.2f} to {self.R_end * 1000:.2f} mm, top speed {self.vv.max() * 1000:.0f} mm/s, swallowed {self.n_eaten}, left {len(self.trail)}", flush=True)

    # ---- where it goes ----------------------------------------------------------------------------
    def _path(self, site):
        m = self.pt["mesh"]
        L, Phi = self.pt["meta"]["L"], self.pt["meta"]["Phi"]
        N = m["P"].shape[0]
        w = np.array(WAY + ((site["i"], site["j"]),), "f8")
        u = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(w, axis=0).T))])
        ij = CubicSpline(u, w)(np.linspace(0, u[-1], 3000))
        P = _bilerp(m["P"], ij).astype("f8")
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        self.L = float(s[-1])
        # even steps, in metres
        S = np.linspace(0, self.L, 1500)
        ij = np.stack([np.interp(S, s, ij[:, 0]), np.interp(S, s, ij[:, 1])], 1)
        P = _bilerp(m["P"], ij).astype("f8")
        ds = S[1] - S[0]
        n = _bilerp(m["n"], ij).astype("f8")
        n = gaussian_filter1d(n, TILT / ds, axis=0, mode="nearest")
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n[n[:, 1] < 0] *= -1
        b = np.gradient(gaussian_filter1d(P, .0005 / ds, axis=0, mode="nearest"), axis=0)
        b -= n * (b * n).sum(1, keepdims=True)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        self.S, self.P, self.n, self.b = S, P, n, b
        self.tb = np.cross(b, n)
        self.ij = ij
        self.w = _bilerp(m["w"], ij) if m["w"].ndim == 2 else (ij[:, 0] / N) ** 2
        r = (ij[:, 0] + .5) / N * L
        self.uv = np.stack([(ij[:, 1] + .5) / N, (ij[:, 0] + .5) / N, 1 / (2 * Phi * np.maximum(r, .002)), np.full(len(r), 1 / L)], 1)
        y = gaussian_filter1d(P[:, 1], .0008 / ds, mode="nearest")
        self.slope = np.clip(-np.gradient(y, ds), 0, 1)
        # the sheet isn't clean: it holds a drop harder across a vein than along one
        phases = np.random.default_rng(1412).uniform(0, 6.283, 4)
        self.rough = 1 + sum(a * np.sin(2 * np.pi * S / lam + p) for a, lam, p in zip((.16, .13, .10, .07), (.0041, .0027, .0017, .0011), phases))
        # what might get swallowed
        self.prey = [(k, a[:, :3].astype("f8"), a[:, 12].astype("f8")) for k, a in ((key, self.fl.bead_arr[key]) for key in (("beads", self.ip), ("micro", self.ip)) if key in self.fl.bead_arr)]

    def _idx(self, s):
        return min(int(s / (self.S[1] - self.S[0]) + .5), len(self.S) - 1)

    # ---- how it goes ------------------------------------------------------------------------------
    def _run(self, drag):
        """the drop, from the moment it lands: arrays of its place, speed and volume, and when its front gets to the lip."""
        dt = self.DT
        n = int((T_OVER + 1.0 - T_LAND) / dt)
        ss, vv, R3 = np.zeros(n), np.zeros(n), np.zeros(n)
        s, v, r3 = 0.0, 0.0, R_LAND ** 3
        eaten = {k: np.full(len(a), np.inf) for k, _, a in self.prey}
        t_front, t_go = None, None
        for k in range(n):
            t = T_LAND + k * dt
            i = self._idx(s)
            R = r3 ** (1 / 3)
            cling = 1 + 1.6 * (1 - float(smooth(T_LAND + .25, T_LAND + .62, t)))
            a = G * max(self.slope[i], SLOPE_MIN) - PIN * (R_LAND / R) ** 2 * cling * self.rough[i]
            if v > 0 or a > 0:
                v = max(v + (a - drag * v) * dt, 0.0)
            if t_go is None and s > .0002:
                t_go = t
            step = v * dt
            s = min(s + step, self.L)
            r3 += FILM * step
            # what it runs into
            q = self.P[i]
            for key, Pb, Rb in self.prey:
                near = (eaten[key] == np.inf) & (np.abs(Pb[:, 1] - q[1]) < R * 1.1 + .001)
                if near.any():
                    idx = np.nonzero(near)[0]
                    hit = idx[np.linalg.norm(Pb[idx] - q, axis=1) < R * .92 + Rb[idx]]
                    if len(hit):
                        eaten[key][hit] = t
                        r0 = r3
                        r3 += float((Rb[hit] ** 3).sum()) * .7
                        v *= r0 / r3        # it has that much more to carry for a moment
            if t_front is None and s >= self.L - R * .9:
                t_front = t
            if s >= self.L:
                v = 0.0
            ss[k], vv[k], R3[k] = s, v, r3
        return ss, vv, R3, eaten, t_front, t_go

    def _simulate(self):
        lo, hi = 15.0, 900.0
        for _ in range(24):
            drag = (lo * hi) ** .5
            ss, vv, R3, eaten, t_front, t_go = self._run(drag)
            if t_front is None or t_front > T_OVER:
                hi = drag
            else:
                lo = drag
        self.drag = drag
        self.tt = T_LAND + np.arange(len(ss)) * self.DT
        self.ss, self.vv, self.R3 = ss, vv, R3
        self.eaten, self.t_front, self.t_go = eaten, t_front, t_go
        self.R_end = float(np.interp(T_OVER, self.tt, R3)) ** (1 / 3)

    def _swallowed(self):
        """the ones it has been through are not drawn after that. they go last in their buffers, in the order it got to them."""
        self.dead = {}
        self.n_eaten = 0
        for key, _, _ in self.prey:
            T = self.eaten[key]
            order = np.argsort(-T, kind="stable")
            self.fl.bead_vbs[key].write(self.fl._bead_inst(self.fl.bead_arr[key][order]).tobytes())
            self.dead[key] = -T[order]
            self.n_eaten += int(np.isfinite(T).sum())

    def alive(self, key, t, n):
        d = self.dead.get(key)
        return n if d is None else int(np.searchsorted(d, -t, side="left"))

    def _trail(self):
        """what doesn't go with it. small, and more of it where it took its time."""
        rng = np.random.default_rng(77)
        out, when = [], []
        i0 = int(np.searchsorted(self.ss, .0015))
        s0, s1 = float(self.ss[i0]), self.L - .0022
        for _ in range(70):
            s = float(rng.uniform(s0, s1))
            k = self._idx(s)
            # when its tail has got past here
            tk = int(np.searchsorted(self.ss, s + .0016))
            if tk >= len(self.tt):
                continue
            Rb = float(np.clip(np.exp(rng.normal(np.log(.00019), .5)), .00007, .00052))
            off = float(rng.normal(0, .00042))
            p = self.P[k] + self.tb[k] * off
            u = self.uv[k]
            out.append([*p, *self.n[k], *self.tb[k], *self.b[k], Rb, rng.uniform(1.0, 1.25), 0, rng.uniform(.15, .42), u[0], u[1], u[2], u[3], self.w[k], 0])
            when.append(float(self.tt[tk]))
        order = np.argsort(when)
        self.trail = np.array(out, "f4")[order]
        self.trail_t = np.array(when)[order]

    def trail_count(self, t):
        return int(np.searchsorted(self.trail_t, t, side="right"))

    # ---- what it is at a moment --------------------------------------------------------------------
    def _x(self, t):
        """how much of it has gone over."""
        return float(smooth(T_OVER - .04, T_OVER + .26, t))

    def at(self, t):
        if t < T_LAND:
            return None
        x = self._x(t)
        s = float(np.interp(t, self.tt, self.ss))
        v = float(np.interp(t, self.tt, self.vv))
        R = float(np.interp(t, self.tt, self.R3)) ** (1 / 3) * (1 - x) ** (1 / 3)
        if R < .00006:
            return None
        u_ = t - T_LAND
        ring = np.exp(-7 * u_) * np.sin(38 * u_)
        R = R * (1 + .22 * ring) * min(u_ / .06 + .3, 1.0)
        S = self.S
        fi = s / (S[1] - S[0])
        i0 = min(int(fi), len(S) - 2)
        f = fi - i0
        lerp = lambda a: a[i0] * (1 - f) + a[i0 + 1] * f
        n, b = lerp(self.n), lerp(self.b)
        tb = lerp(self.tb)
        uv = lerp(self.uv)
        e = 1.0 + min(v / .032, 1.0) * .42
        # it empties toward where the water is going: under the edge, where the neck is
        site = self.fl.drop.site
        return dict(P=lerp(self.P) + site["inward"] * .0007 * x, n=n, t=tb, b=b, R=R, e=e, zc=.30, u=uv[0], v=uv[1], dudm=uv[2], dvdm=uv[3], w=float(lerp(self.w)), free=False)

    def view(self, t):
        """where it is as the camera sees it. once it has gone over, that is the drop."""
        d = self.fl.drop
        s = float(np.interp(t, self.tt, self.ss))
        P = self.P[self._idx(s)]
        tan = d.tan
        a = (float(P[0] / P[2] / tan), float(P[1] / P[2] / tan), float(P[2]))
        k = float(smooth(T_OVER, T_OVER + .55, t))
        return tuple(a[i] + ((*d.view, d.z)[i] - a[i]) * k for i in range(3))

    def focus(self, t):
        """the plane AF-C holds while it follows this: from the drop on the sheet to the skin of the one that hangs."""
        if t < T_LAND or t >= 11.3:
            return None
        a = 1 / self.view(min(t, T_OVER))[2]
        return a + (self.fl.drop.f_surface - a) * float(smooth(T_OVER, T_OVER + .6, t))

    def swell(self, t):
        """how big what hangs at the edge is, as a fraction of how big it ends up."""
        v = SHARE * self._x(t) + (1 - SHARE) * float(smooth(T_OVER + .08, T_OVER + .95, t))
        return v ** (1 / 3)

    def ring(self, t):
        """(stretch, swing). it arrives with its way on: it goes on past where it will hang, and comes back, and goes on."""
        s = self.swell(t)
        st = -.40 * (1 - float(smooth(.30, .95, s))) if s > 0 else 0.0      # a small one has too little weight to have a neck: it is a cap
        u = t - T_OVER - .10
        if u <= 0:
            return st, 0.0
        st += .13 * np.exp(-2.5 * u) * np.sin(2 * np.pi * 9.0 * u)
        return st, .135 * np.exp(-1.45 * u) * np.sin(2 * np.pi * 4.3 * u)
