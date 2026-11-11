"""A loose flock of sea birds: a real boids simulation, integrated once with a fixed seed.

The result is a table indexed by film time, so any frame can be evaluated on its own.
Coordinates: metres, camera at the origin, +z towards the sun, +x right, +y up.
"""
from __future__ import annotations
import numpy as np

N_BIRDS = 24
RATE = 30.0
T0, T1 = -14.0, 130.0


def _simulate(seed=71):
    rng = np.random.default_rng(seed)
    n = int((T1 - T0) * RATE) + 1
    pos = np.zeros((n, N_BIRDS, 3), "f4")
    p = np.array([-190.0 + 11.0 * (T0 - 18.0), 24.0, 700.0]) + rng.normal(size=(N_BIRDS, 3)) * np.array([26.0, 3.0, 26.0])
    v = np.tile(np.array([11.0, 0.0, -1.0]), (N_BIRDS, 1)) + rng.normal(size=(N_BIRDS, 3)) * .6
    dt = 1 / RATE
    for i in range(n):
        pos[i] = p
        t = T0 + i * dt
        d = p[:, None, :] - p[None, :, :]
        r = np.linalg.norm(d, axis=2) + np.eye(N_BIRDS) * 1e6
        near = r < 26.0
        cnt = near.sum(1, keepdims=True).clip(1)
        coh = ((p[None, :, :] * near[..., None]).sum(1) / cnt - p) * near.any(1, keepdims=True)
        ali = ((v[None, :, :] * near[..., None]).sum(1) / cnt - v) * near.any(1, keepdims=True)
        sep = (d / (r[..., None] ** 2) * (r < 8.0)[..., None]).sum(1)
        # the whole flock follows a slow meandering line, low over the water
        goal = np.array([-190.0 + 11.0 * (t - 18.0), 22.0 + 5.0 * np.sin(t * .13), 700.0 + 60.0 * np.sin(t * .071 + 1.0)])
        home = goal - p
        a = .22 * coh / 8 + .9 * ali + 60.0 * sep + .06 * home + rng.normal(size=(N_BIRDS, 3)) * np.array([.5, .28, .5])
        a[:, 1] -= .25 * v[:, 1]
        v = v + a * dt
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / sp * np.clip(sp, 8.5, 13.5)
        p = p + v * dt
    flap_rate = rng.uniform(2.6, 3.5, N_BIRDS)
    flap_phase = rng.uniform(0, 6.28, N_BIRDS)
    glide_rate = rng.uniform(.14, .26, N_BIRDS)
    glide_phase = rng.uniform(0, 6.28, N_BIRDS)
    span = rng.uniform(1.15, 1.75, N_BIRDS)
    return pos, flap_rate, flap_phase, glide_rate, glide_phase, span


class Flock:
    def __init__(self):
        self.pos, self.fr, self.fp, self.gr, self.gp, self.span = _simulate()
        self.n = N_BIRDS

    def at(self, t):
        """-> (az, el, half-span in rad, wing lift), (heading roll, unused) per bird."""
        x = float(np.clip((t - T0) * RATE, 0, len(self.pos) - 1.001))
        i = int(x)
        f = x - i
        p = self.pos[i] * (1 - f) + self.pos[i + 1] * f
        r = np.linalg.norm(p, axis=1)
        az = np.arctan2(p[:, 0], p[:, 2])
        el = np.arcsin(p[:, 1] / r)
        half = self.span / r
        glide = np.clip(.5 + 1.1 * np.sin(self.gr * 6.283 * t + self.gp), 0, 1)
        wing = .12 + glide * .62 * np.sin(self.fr * 6.283 * t + self.fp)
        return np.stack([az, el, half, wing], 1).astype("f4")
