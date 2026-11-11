"""The whole light path after the world: camera body -> EVF OLED -> eyepiece -> phone."""
from __future__ import annotations
import numpy as np
import moderngl
from .core import GL, Target

PANEL = (1024, 768)
BARS = 96 / 768


def make_dust(gl: GL, seed=3):
    rng = np.random.default_rng(seed)
    w, h = 512, 384
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.ones((h, w), "f4")
    for _ in range(26):                     # soft specks: out of focus, ring-ish
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(.8, 3.2)
        a = rng.uniform(.04, .22)
        dd = np.hypot(xx - cx, yy - cy)
        d *= 1 - a * np.exp(-((dd - r * .7) / (r * .55)) ** 2)
    t = gl.ctx.texture((w, h), 1, d.astype("f4").tobytes(), dtype="f4")
    t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t.repeat_x = t.repeat_y = False
    return t


class Optics:
    def __init__(self, gl: GL, out=(1920, 1080), sim=2):
        self.gl = gl
        self.out = out
        self.sim_size = (out[0] * sim, out[1] * sim)
        self.panel = gl.target(PANEL, dtype="f1", mips=True)
        self.raw = gl.target(self.sim_size, comps=1, dtype="f2", nearest=True)
        self.rgb = gl.target(self.sim_size, dtype="f2", mips=True)
        self.hist = [gl.target(self.sim_size, dtype="f2", mips=True) for _ in range(2)]
        self.final = gl.target(out, dtype="f1")
        self.hud_tex = gl.ctx.texture(PANEL, 4, dtype="f1")
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_tex.repeat_x = self.hud_tex.repeat_y = False
        self.rows = gl.ctx.texture((1, 64), 2, dtype="f4")
        self.rows.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.rows.repeat_x = self.rows.repeat_y = False
        self.dust = make_dust(gl)
        self.gain = None
        self.flip = 0
        self.fresh = True

    def body(self, world: Target, hud_rgba, st: dict, frame: int, over: float, into=None):
        if hud_rgba is not None:
            self.hud_tex.write(np.flipud(hud_rgba).tobytes())
        target = into if into is not None else self.panel
        self.gl.run(
            "isp.frag", target, uWorld=world, uHud=self.hud_tex, uEV=st.get("ev", 0.0),
            uISO=st.get("iso", 1.0), uFrame=float(frame % 4096), uZebra=st.get("zebra", 0.0),
            uPeak=st.get("peak", 0.0), uPeakCol=st.get("peak_col", (1.0, .15, .1)),
            uRoll=st.get("roll_small", 0.0), uRS=st.get("rs", 0.0), uOver=over,
            uSat=st.get("sat", 1.0), uShift=st.get("ibis", (0.0, 0.0)), uBars=BARS,
            uPassthrough=st.get("passthrough", 0.0),
            uBgDisp=st.get("bg_disp", self.hud_tex), uHasBgDisp=1.0 if "bg_disp" in st else 0.0,
        )
        target.gen()

    def meter(self):
        b = np.frombuffer(self.panel.tex.read(level=6, alignment=1), "u1")
        w, h = max(PANEL[0] >> 6, 1), max(PANEL[1] >> 6, 1)
        a = b.reshape(h, w, 4)[..., :3].astype("f4") / 255
        return float(np.mean(a ** 2.2)), float(np.percentile(a ** 2.2, 97))

    def phone(self, ph: dict, hands, t: float, frame: int, dt=1 / 30):
        mean, hi = self.meter()
        # AE: chases the frame with lag; black surround means it tends to run hot
        tgt = np.clip(ph.get("ae_target", .10) / max(mean, 1e-3), ph.get("gain_min", .35), ph.get("gain_max", 9.0))
        if self.gain is None:
            self.gain = tgt
        tau = ph.get("ae_tau", .55)
        self.gain += (tgt - self.gain) * (1 - np.exp(-dt / tau))
        if "gain_lock" in ph:
            self.gain = ph["gain_lock"]
        gain = self.gain * ph.get("gain_mul", 1.0)
        self.rows.write(hands.rows(t).tobytes())
        h = hands.at(t)
        scale = ph.get("scale", .985) * (1 + h["dz"] * ph.get("shake", 1))
        center = (ph.get("cx", 0.0) + h["dx"], ph.get("cy", 0.0) + h["dy"])
        pupil = (ph.get("px", 0.0) + h["dx"] * 3.5 + h["tx"] * 1.2, ph.get("py", 0.0) + h["dy"] * 3.5 + h["ty"] * 1.2)
        self.gl.run(
            "phone_sensor.frag", self.raw, uPanel=self.panel, uRows=self.rows, uDust=self.dust,
            uRes=self.sim_size, uPanelC=center, uPupil=pupil, uTilt=(h["tx"] * .8, h["ty"] * .8),
            uPanelScale=scale, uRoll=h["droll"] + ph.get("roll", 0.0), uK1=ph.get("k1", .055), uK2=.03,
            uCA=ph.get("ca", .0035), uDefocus=ph.get("defocus", .55), uFC=ph.get("fc", 2.1),
            uGain=gain, uFrame=float(frame % 8192), uMask=ph.get("mask", .42),
            uPupilR=ph.get("pupil_r", .86), uPupilSoft=ph.get("pupil_soft", .075),
            uBandDepth=ph.get("band", .018), uBandCount=ph.get("band_n", 1.6),
            uBandPhase=t * ph.get("band_speed", .17), uVeil=ph.get("veil", .022),
            uGhost=ph.get("ghost", .012), uRubber=ph.get("rubber", .07), uPSF=ph.get("psf", 1.0),
            uNoise=ph.get("noise", 1.0), uDustK=ph.get("dust", 1.0), uFlat=ph.get("flat", 0.0),
        )
        self.gl.run("demosaic.frag", self.rgb, uRaw=self.raw)
        self.rgb.gen()
        cur, prev = self.hist[self.flip], self.hist[1 - self.flip]
        sigma = .012 * max(gain, .3) ** .5
        self.gl.run("tnr.frag", cur, uCur=self.rgb, uPrev=prev, uStrength=ph.get("tnr", .62),
                    uSigma=sigma, uReset=1.0 if self.fresh else 0.0)
        cur.gen()
        self.fresh = False
        self.flip = 1 - self.flip
        px = (1 / self.sim_size[0], 1 / self.sim_size[1])
        chroma = float(np.clip(1.2 + np.log2(max(gain, 1)) * .7, 1.2, 3.4))
        self.gl.run("finish.frag", self.final, uIn=cur, uSimPx=px, uSharp=ph.get("sharp", .72),
                    uChromaLod=chroma, uShadow=ph.get("expo", 1.0), uFrame=float(frame % 4096))
        return gain

    def read(self) -> np.ndarray:
        data = self.final.fbo.read(components=3, dtype="f1", alignment=1)
        return np.flipud(np.frombuffer(data, "u1").reshape(self.out[1], self.out[0], 3))
