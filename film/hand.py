"""A simulated pair of hands: operator holds a camera, and a phone to its eyepiece.

Everything is a spring-damper driven by physiological noise bands (drift, breath,
pulse, tremor). The strength knobs are ordinary numbers that the story can animate.
"""
from __future__ import annotations
import numpy as np


def _band(rng, n, dt, freqs, amps):
    t = np.arange(n) * dt
    s = np.zeros(n)
    for f, a in zip(freqs, amps):
        ph = rng.uniform(0, 2 * np.pi)
        wob = rng.uniform(.75, 1.25)
        s += a * np.sin(2 * np.pi * f * wob * t + ph + .6 * np.sin(2 * np.pi * f * .13 * t + rng.uniform(0, 6)))
    return s


def _spring(x, dt, hz, zeta):
    w = 2 * np.pi * hz
    y = np.zeros_like(x)
    p, v = x[0], 0.0
    for i in range(len(x)):
        a = w * w * (x[i] - p) - 2 * zeta * w * v
        v += a * dt
        p += v * dt
        y[i] = p
    return y


class Hands:
    RATE = 240

    def __init__(self, duration, seed=4, tremor=lambda t: 1.0, breath=lambda t: 1.0):
        dt = 1 / self.RATE
        n = int(duration * self.RATE) + 8
        rng = np.random.default_rng(seed)
        self.t = np.arange(n) * dt
        tr = np.array([tremor(x) for x in self.t[::24]])
        tr = np.interp(self.t, self.t[::24], tr)
        br = np.array([breath(x) for x in self.t[::24]])
        br = np.interp(self.t, self.t[::24], br)

        def layer(drift, breath_, pulse, trem):
            return (
                _band(rng, n, dt, [.07, .13, .23, .41], np.array([1, .8, .55, .35]) * drift) * tr
                + _band(rng, n, dt, [.27], [breath_]) * br
                + _band(rng, n, dt, [1.15, 2.3], [pulse, pulse * .35]) * tr
                + _band(rng, n, dt, [8.3, 9.7, 11.2, 13.1], np.array([1, .8, .6, .4]) * trem) * tr
            )

        # camera-to-world (radians), after the body's IBIS has eaten most of the tremor
        self.yaw = _spring(layer(.0016, .0009, .00022, .00006), dt, 7, .55)
        self.pitch = _spring(layer(.0014, .0013, .00025, .00006), dt, 7, .55)
        self.roll = _spring(layer(.0026, .0012, .0002, .00005), dt, 6, .6)
        # phone relative to eyepiece (frame-height units), less IBIS -> more jello
        self.dx = _spring(layer(.0052, .0018, .0005, .00034), dt, 14, .4)
        self.dy = _spring(layer(.0046, .0024, .00055, .00034), dt, 14, .4)
        self.dz = _spring(layer(.0040, .0012, .0003, .00012), dt, 9, .5)
        self.droll = _spring(layer(.0022, .0008, .0002, .00016), dt, 9, .5)
        self.tx = _spring(layer(.010, .0035, .0008, .0002), dt, 9, .5)
        self.ty = _spring(layer(.010, .0045, .0008, .0002), dt, 9, .5)

    def at(self, t):
        i = np.clip(t * self.RATE, 0, len(self.t) - 2)
        a = int(i)
        f = i - a
        lerp = lambda s: s[a] * (1 - f) + s[a + 1] * f
        return {k: lerp(getattr(self, k)) for k in
                ("yaw", "pitch", "roll", "dx", "dy", "dz", "droll", "tx", "ty")}

    def rows(self, t, n=64, readout=1 / 40):
        """Phone shake across the rolling-shutter readout of a single frame."""
        ts = t + np.linspace(0, readout, n)
        i = np.clip(ts * self.RATE, 0, len(self.t) - 2).astype(int)
        base = self.at(t)
        return np.stack([self.dx[i] - base["dx"], self.dy[i] - base["dy"]], axis=1).astype("f4")
