"""Ink-wash handscroll: the camera's physics translated into the physics of ink.

A couplet and a seal are brushed on the sheet at scroll x 1.77..2.06, above the sun once pan is about 1.76: 落日一舟, 万山皆字.
A zoom centred on its last character resolves it into thousands of 山.

World state (`st`, all optional except t):
  t           seconds; only the mist drifts with it.
  pan         position along the scroll in view units, positive looks right. Layers have parallax.
  pan_speed   view units/s, signed. Above about 1 u/s the pan becomes feibai (flying white).
  blur        0..1 focus error: wet-in-wet diffusion over the paper's fibre conductivity. The iteration
              restarts from the sharp painting each frame; the count is proportional to blur.
  zoom        >=1 magnification about zc. Zoom 1 is the identity whatever zc is; a point v in the unzoomed
              view lands on p = (v - zc*(1-1/zoom)) * zoom. Beyond about 12 the ink resolves into characters.
  zc          (x, y) zoom centre in unzoomed view units.
  from        optional Target (linear light): the shot being left behind.
  reveal      0..1 optional, transition out of `from` (only used when `from` is given).
  hist        truthy: read back a small copy of the frame and keep its 64-bin luma histogram in self.hist.
              (hud["hist"] = True makes draw_hud use it. Its window is HIST_RANGE, the paper's tones.)

Helpers for whoever aims the AF box: boat_figure(st) gives the tiny rectangle in view units, view_to_image(v, st)
turns view units into the 0..1 image coordinates that s["af"] wants.
"""
from __future__ import annotations
import numpy as np
import moderngl
from PIL import Image
from ..core import GL, Target, Track, smooth
from .. import hud_ink
from . import ink_scroll as S
from .ink_tex import PAPER_P, tex2d, ATLAS_RES

WORLD_OVER = 1.06
CELL = 2.0 / (60.0 * 11.0)     # about 11 characters high fill the frame at zoom 60
ATLAS_UNIT = 15

HIST_RANGE = (.822, .885)      # the histogram's window: the paper's tones, where a shan-shui keeps its pixels
N_MAX = 360                    # diffusion iterations at blur=1
STEPS = (1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 4.0)     # stencil radii, a fixed cycle


def eased(r, a, b):
    return float(smooth(a, b, r))


_LD_PAN = Track([(30.0, 0.0), (34.0, 1.2), (35.0, 4.4)])
_LD_BLUR = Track([(35.0, 0.0), (35.75, .8), (36.5, 0.0)])
_LD_ZC = (-.40, .43)           # a ridge of the twin peaks that the quick pan lands on
_LD_AFX = Track([(29.0, .55), (30.0, .36), (30.9, .50), (34.0, .50), (34.6, .22), (35.2, .5), (36.6, .5), (38.0, .5)])
_LD_AFY = Track([(29.0, .46), (30.0, .60), (30.9, .66), (34.0, .66), (34.6, .50), (35.2, .5), (36.6, .46), (38.0, .5)])
_LD_AFS = Track([(29.0, 90.0), (30.0, 50.0), (30.9, 40.0), (34.0, 40.0), (34.6, 70.0), (35.2, 130.0), (36.6, 60.0), (38.0, 34.0)])


class Ink:
    def __init__(self, gl: GL, size, log=print, rebake=()):
        self.gl = gl
        self.size = tuple(size)
        self.aspect = size[0] / size[1]
        a = S.ensure(gl, log=log, rebake=rebake)
        self.t = {}
        for k in ("far", "hill", "mid", "water", "near", "fore"):
            t = tex2d(gl, np.asarray(a[k]))
            t.repeat_y = False
            self.t[k] = t
        self.t["noise"] = tex2d(gl, a["noise"])
        self.t["paper"] = tex2d(gl, a["paper"])
        self.t["K"] = tex2d(gl, a["K"])
        self.t["mist"] = tex2d(gl, a["mist"])
        self.t["sun"] = tex2d(gl, a["sun"], repeat=False)
        self.t["boat"] = tex2d(gl, a["boat"], repeat=False)
        carved = hud_ink.get()._carve(S.SEAL["ch"], 48, -1.4)      # same knife as the camera's seal
        self.seal_size = S.SEAL["size"] * carved.shape[0] / (48 * hud_ink.S)
        self.t["seal"] = tex2d(gl, (np.sqrt(np.clip(carved / S.DMAX, 0, 1))[::-1] * 255 + .5).astype("u1"), repeat=False)
        at = a["atlas"]
        self.atlas = gl.ctx.texture_array((at.shape[2], at.shape[1], at.shape[0]), 1, at.tobytes(), dtype="f1")
        self.atlas.build_mipmaps()
        self.atlas.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.atlas.repeat_x = self.atlas.repeat_y = False
        self.atlas_mean = tuple(float(x) for x in a["atlas_mean"])
        self.t_paint = gl.target(size)
        self.t_fb = gl.target(size)
        self.t_fb0 = gl.target(size)
        self.t_s = [gl.target(size) for _ in range(2)]
        self.t_dummy = None
        self.small = gl.target((212, 119), dtype="f2")
        self.hist = None
        self._paper_done = False

    # ---- where things are ----------------------------------------------------------------
    def view_of(self, X, rate, period, st):
        """Scroll position (at pan 0) -> unzoomed view x at this pan, wrapped to the visible period."""
        return ((X - st.get("pan", 0.0) * rate + period / 2) % period) - period / 2

    def boat_figure(self, st=None):
        """The small rectangle held up by the figure, in unzoomed view units at this pan."""
        st = st or {}
        x = self.view_of(S.BOAT[0], S.BOAT_RATE, S.LAYERS["water"]["P"], st) + S.BOAT_FIGURE_OFF[0]
        return (x, S.BOAT[1] + S.BOAT_FIGURE_OFF[1])

    def view_to_image(self, v, st=None):
        """Unzoomed view coordinates -> image coords (0..1, y down) after zoom and the overscan crop, for the AF box."""
        st = st or {}
        z = float(st.get("zoom", 1.0))
        zc = st.get("zc", (0.0, 0.0))
        px = (v[0] - zc[0] * (1 - 1 / z)) * z
        py = (v[1] - zc[1] * (1 - 1 / z)) * z
        return (.5 + px / (2 * self.aspect / WORLD_OVER), .5 - py / (2 / WORLD_OVER))

    # ---- one frame ------------------------------------------------------------------------
    def render(self, st: dict, out: Target):
        gl = self.gl
        t = float(st["t"])
        pan = float(st.get("pan", 0.0))
        speed = float(st.get("pan_speed", 0.0))
        blur = float(np.clip(st.get("blur", 0.0), 0.0, 1.0))
        zoom = float(max(st.get("zoom", 1.0), 1.0))
        zc = tuple(float(x) for x in st.get("zc", (0.0, 0.0)))
        src = st.get("from")
        r = float(np.clip(st.get("reveal", 1.0), 0.0, 1.0)) if src is not None else 1.0
        L = S.LAYERS
        ph = lambda k: float((pan * L[k]["rate"]) % L[k]["P"])
        sx = ((S.SUN[0] - pan * S.SUN_RATE + 40) % 80) - 40
        bx = ((S.BOAT[0] - pan * S.BOAT_RATE + 8) % 16) - 8
        s = S.SUN_SPRITE
        # storyboard of the arrival
        rg = eased(r, .30, .86) if r < 1 else 1.0
        rev = (eased(r, .46, .92), eased(r, .36, .78), eased(r, .52, .96), eased(r, .62, 1.0)) if r < 1 else (1.0,) * 4
        sun_a = eased(r, .12, .66) if r < 1 else 1.0
        kred = (.15 * sun_a, 1.22 * sun_a ** 1.6, 1.62 * sun_a ** .55)
        gl.run("ink_paint.frag", self.t_paint, uFar=self.t["far"], uHill=self.t["hill"], uMid=self.t["mid"], uWater=self.t["water"],
               uNear=self.t["near"], uFore=self.t["fore"], uSunT=self.t["sun"], uBoat=self.t["boat"], uMist=self.t["mist"],
               uSeal=self.t["seal"], uSealC=S.SEAL["c"], uSealS=self.seal_size,
               uAspect=self.aspect, uZoom=zoom, uZc=zc, uT=t, uDmax=S.DMAX, uLodBias=.45, uHorizon=S.HORIZON,
               uPan=float(pan % (L["water"]["P"] * 40)),
               uPh=(ph("far"), ph("hill"), ph("mid"), ph("near")), uPer=(L["far"]["P"], L["hill"]["P"], L["mid"]["P"], L["near"]["P"]),
               uPhFore=ph("fore"), uPerFore=L["fore"]["P"], uPerWater=L["water"]["P"],
               uSunC=(sx, S.SUN[1]), uSunRect=(s["x0"], s["x1"], s["y0"], s["y1"]), uBoatC=(bx, S.BOAT[1]), uBoatSize=S.BOAT_SPRITE["size"],
               uRG=rg, uRev=rev)
        d = self.t_paint
        phase = float(pan % PAPER_P)

        # ---- pan -> flying white
        amt = float(smooth(.5, 2.6, abs(speed)))
        if amt > 0:
            kw = dict(uNoise=self.t["noise"], uAspect=self.aspect, uZoom=zoom, uZc=zc, uPhPaper=phase,
                      uDir=float(np.sign(speed)), uLen=min(abs(speed) * .15, 1.1) / zoom, uAmt=amt)
            gl.run("ink_feibai.frag", self.t_fb0, uD=d, uStage=0.0, **kw)
            gl.run("ink_feibai.frag", self.t_fb, uD=self.t_fb0, uStage=1.0, **kw)
            d = self.t_fb

        # ---- focus error -> wet-in-wet
        if blur > 0:
            exact = blur * N_MAX
            n = int(np.ceil(exact))
            gl.run("ink_kmap.frag", self.t_s[0], uD=d, uK=self.t["K"], uAspect=self.aspect, uZoom=zoom, uZc=zc, uPhPaper=phase, uPaperP=PAPER_P)
            px = (1 / self.size[0], 1 / self.size[1])
            for i in range(n):
                dt = .115 * (1.0 if i < n - 1 else (exact - (n - 1)))
                gl.run("ink_diffuse.frag", self.t_s[(i + 1) % 2], uS=self.t_s[i % 2], uPx=px, uStep=STEPS[i % len(STEPS)], uDt=dt, uMob=.10)
            d = self.t_s[n % 2]

        # ---- colour, characters, the incoming sea
        self.atlas.use(ATLAS_UNIT)
        kw = dict(uHasFrom=0.0, uReveal=1.0)
        if src is not None and r < 1:
            kw = dict(uHasFrom=1.0, uReveal=r, uFrom=src)
        self._comp(out, d, zoom, zc, phase, pan, sx, kred, smooth(.72, 1.0, r), kw)
        if st.get("hist"):
            self.hist = self.histogram(out)

    def _comp(self, out, d, zoom, zc, phase, pan, sx, kred, settle, kw):
        self.atlas.use(ATLAS_UNIT)
        self.gl.run("ink_comp.frag", out, uD=d, uReg=self.t_paint, uPaper=self.t["paper"], uNoise=self.t["noise"], uMist=self.t["mist"],
                    uAspect=self.aspect, uZoom=zoom, uZc=zc, uPhPaper=phase, uPaperP=PAPER_P, uCell=CELL, uPhGlyph=float(pan % (CELL * 2048)),
                    uAtlasRes=float(ATLAS_RES), uMean=self.atlas_mean, uAtlas=ATLAS_UNIT, uHorizon=S.HORIZON, uSunR=S.SUN_R, uSunC=(sx, S.SUN[1]),
                    uSettle=float(settle), uKRed=kred, **kw)

    def histogram(self, frame: Target):
        """The real 64-bin luma histogram of the frame, from a small point-sampled copy, over the paper's tones."""
        self.gl.run("ink_small.frag", self.small, uTex=frame)
        w, h = self.small.size
        a = np.frombuffer(self.small.fbo.read(components=3, dtype="f2", alignment=1), "f2").reshape(h, w, 3).astype("f4")
        l = a @ np.array([.2126, .7152, .0722], "f4")
        n, _ = np.histogram(l, bins=64, range=HIST_RANGE)
        return n.astype("f4")

    # ---- the handwritten display -----------------------------------------------------------------
    def _paper_strip(self):
        """The bars are cut from the world's own sheet: its paper, run through the same shader with no ink on it,
        mirrored at the seam so that the tone is continuous across the edge of the image."""
        gl = self.gl
        tmp = gl.target(self.size)
        zero = gl.target(self.size)
        zero.clear()
        self.t_paint.clear()
        self._comp(tmp, zero, 1.0, (0.0, 0.0), 0.0, 0.0, 40.0, (0.0, 0.0, 0.0), 1.0, dict(uHasFrom=0.0, uReveal=1.0))
        w, h = self.size
        a = np.frombuffer(tmp.fbo.read(components=3, dtype="f2", alignment=1), "f2").reshape(h, w, 3).astype("f4")[::-1]
        cx, cy = (w - w / WORLD_OVER) / 2, (h - h / WORLD_OVER) / 2
        box = (cx, cy, w - cx, h - cy)
        image = np.stack([np.asarray(Image.fromarray(a[..., i], "F").resize((1024, 576), Image.BILINEAR, box=box)) for i in range(3)], -1)
        panel = np.zeros((768, 1024, 3), "f4")
        panel[96:672] = image
        panel[:96] = image[:96][::-1]
        panel[672:] = image[-96:][::-1]
        for t in (tmp, zero):
            t.fbo.release()
            t.tex.release()
        return panel

    def draw_hud(self, s: dict):
        hud = hud_ink.get()
        if not self._paper_done:
            hud.set_paper(self._paper_strip())
            self._paper_done = True
        if s.get("hist") is True:
            # the window is the paper's tones, so the silhouette rises as the paper arrives
            share = 0.0 if self.hist is None else float(self.hist.sum()) / (self.small.size[0] * self.small.size[1])
            s = {**s, "hist": self.hist, "hist_k": float(smooth(.02, .30, share))}
        return hud.draw(s)

    # ---- the look-development reel ----------------------------------------------------------------
    def lookdev(self, t: float) -> dict:
        t = float(t)
        pan = float(_LD_PAN(t))
        speed = float((_LD_PAN(t + .005) - _LD_PAN(t - .005)) / .01)
        zoom = float(60.0 ** smooth(36.5, 40.0, t))
        st = dict(name="ink", t=t, pan=pan, pan_speed=speed, blur=float(_LD_BLUR(t)), zoom=zoom, zc=_LD_ZC, hist=True)
        if t < 30.0:
            st["reveal"] = float(np.clip((t - 28.0) / 2.0, 0.0, 1.0))
            st["from"] = self.dummy_sea(t)
        # AF: looks about, finds the boat and stays on the figure, is thrown by the pan, gives up in the wet
        fig = self.view_to_image(self.boat_figure(st), st)
        boat = smooth(30.9, 31.5, t) * (1 - smooth(33.7, 34.0, t))
        ax = float(_LD_AFX(t)) * (1 - boat) + fig[0] * boat
        ay = float(_LD_AFY(t)) * (1 - boat) + fig[1] * boat
        size = float(_LD_AFS(t)) * (1 - boat) + 30 * boat
        state = "fail" if 34.0 <= t < 35.2 else "dim" if 35.2 <= t < 36.5 else "lock" if 31.8 <= t < 33.7 else "search"
        af = dict(x=ax, y=ay, w=size, h=size * .86, state=state, alpha=float(smooth(28.6, 29.4, t) * (1 - .6 * smooth(38.6, 39.4, t))))
        hud = dict(shutter="1/50", fnum="F1.8", iso="6400", ev="±0.0", remain="2h 14m", tc="0:00:00", battery=84, afm="AF-C", look="墨",
                   focus="lock" if state == "lock" else "hunt" if state in ("fail", "dim") else "off",
                   ev_meter=float(.12 * np.sin(t * .9)), audio=(float(.28 + .08 * np.sin(t * 2.3)), float(.25 + .09 * np.sin(t * 1.7 + 1))),
                   lens="55mm", timecode="00:00:00.00", wb="AWB", af=af, hist=True)
        if zoom > 1.02:
            del hud["hist"]             # the navigator sits where the histogram sits
            hud["mag"] = dict(x=float(.5 + _LD_ZC[0] / (2 * self.aspect)), y=float(.5 - _LD_ZC[1] / 2), zoom=zoom, label=f"x{zoom:.0f}")
        return dict(world=st, isp=dict(passthrough=1.0, ev=0.0, iso=0.0), hud=hud, ph=dict(defocus=.5, band=.012, ghost=.0, gain_min=.60))

    # ---- the stand-in for the shot before this one ------------------------------------------
    def dummy_sea(self, t):
        if self.t_dummy is None:
            self.t_dummy = self.gl.target(self.size)
        self.gl.run("ink_dummy.frag", self.t_dummy, uAspect=self.aspect, uT=float(t), uHorizon=S.HORIZON, uSunR=S.SUN_R, uSunC=S.SUN, uNoise=self.t["noise"])
        return self.t_dummy
