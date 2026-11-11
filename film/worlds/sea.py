"""A black-sand beach, the second shot of the film: it has to pass as real footage.

Near water: a shallow-water solve over the beach in a log-polar grid around the camera (beach_sim).
Far water: a five-cascade Tessendorf FFT evaluated exactly at t in numpy, filtered with LEAN second
moments, statistical at the horizon; the same short waves ride on the simulated flow.
Sky: an atmosphere and cloud layers raymarched once into panoramas (cached), then only drifted.
All of it is a pure function of st, so any frame can be rendered on its own.
"""
from __future__ import annotations
import numpy as np
import moderngl
from ..core import GL, Target, Track
from .sea_ocean import Ocean, CASCADES
from .sea_birds import Flock
from . import beach_sim as bs
from .sea_sky import SkyBake, SKY, LAYER_ALT, AZ_RANGE, EL_RANGE
from .beach_sim import BeachSim
from .beach_lace import bake_lace, TILE, TILE_READABLE
from .beach_stacks import AZ as STACK_AZ, EL as STACK_EL, RE_EFF, _STACKS, bake_stacks

CAM_H = bs.CAM_H              # metres above mean water, on the back of the beach
AIR_BETA = .18                # km^-1 of near-surface haze seen over the water

# The figure on the skerry. Exported: the AF box in the timeline can pause on it.
FIGURE_AZ = float(np.arctan2(bs.SKERRY[0], bs.SKERRY[1]))
FIGURE_R = float(np.hypot(bs.SKERRY[0], bs.SKERRY[1]))
FIGURE_FOOT = bs.SKERRY[5] + .12         # top of the rock under the feet, m above mean water
FIGURE_H = .0055                         # rad, 9.2 world px at tan_half=.27
FIGURE_EL = float((FIGURE_FOOT - CAM_H) / FIGURE_R)

SHOT_TAN = .27                # the lens of the shot itself
# where in the pixel the looks of a frame land
_AA = ((-.0625, -.1875), (.0625, .1875), (.3125, -.0625), (-.1875, .3125), (-.3125, -.3125), (.1875, -.4375), (-.4375, .0625), (.4375, .4375))
SUN_GAIN = 4.6                # the sun against the sky: a camera holds this ratio, the single-scatter bake does not
HORIZON_DIP = float(np.sqrt(2 * CAM_H / RE_EFF))

def _basis(yaw, pitch, roll):
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    F = np.array([sy * cp, sp, cy * cp])
    R = np.array([cy, 0.0, -sy])
    U = np.cross(F, R)
    cr, sr = np.cos(roll), np.sin(roll)
    R2 = cr * R + sr * U
    U2 = -sr * R + cr * U
    return tuple(float(x) for x in F), tuple(float(x) for x in R2), tuple(float(x) for x in U2)


# ---- lookdev camera
_YAW = Track([(16, -.010), (17.3, .006), (18.4, .002), (19.6, .014), (20.8, .007), (22, .012)])
_PITCH = Track([(16, -.096), (18, -.093), (20, -.097), (22, -.095)])
_WHIP = Track([(22.0, .0), (22.58, .775), (22.95, .75)])
_YAW_LATE = Track([(24, .75), (25.5, .755), (27, .75), (29, .752)])


class Sea:
    TILE_AZ = np.radians(5.0)
    TILE_EL = np.radians(2.5)
    BLOCKS = (12, 24)         # tiles
    SUB = 16                  # texels per tile, 4x4 coarse blocks of 4x4 texels

    def __init__(self, gl: GL, size):
        self.gl = gl
        self.size = tuple(size)
        self.aspect = size[0] / size[1]
        ctx = gl.ctx
        self.ocean = Ocean()
        self.sky = SkyBake(gl)
        self.flock = Flock()
        self.sim = BeachSim(gl)
        # the solver's state in a form the renderer can filter
        NS = self.sim.size
        self.r0 = gl.target(NS, mips=True)
        self.r1 = gl.target(NS, mips=True)
        self.r0s = gl.target(NS, mips=True)     # r0 with the ends of bore sections eased, for the hero views
        self.rfbo = ctx.framebuffer([self.r0.tex, self.r1.tex])
        self.block = 8
        self.hmax = gl.target((NS[0], NS[1] // self.block + 1), comps=1, nearest=True)
        self._t_view = None
        self._bake_terrain()
        self.lace, self.lace_n, self.lace_th = bake_lace(gl)
        self.lace_h, self.lace_nh, self.lace_thh = bake_lace(gl, readable=True)
        self.clouds_h = self.sky.hero_clouds()
        self._hero_now = False
        self.stacks_h = bake_stacks(gl, self.sky.bg, self.sky.sun_dir, tuple(self.sky.sun_e), .5, frag="beach_stacks_hero.frag")
        N = CASCADES[0][1]
        n = len(CASCADES)
        self.tA = ctx.texture_array((N, N, n), 4, dtype="f2")
        self.tB = ctx.texture_array((N, N, n), 4, dtype="f2")
        for t in (self.tA, self.tB):
            t.repeat_x = t.repeat_y = True
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.anisotropy = 16.0
        self.cas = []
        for (L, N_, _, _, rot, _) in CASCADES:
            c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
            self.cas.append((float(c / L), float(s / L), float(L / N_), 0.0))
        self._t_ocean = None
        self.ocean_hmax = float(self.ocean.sigma_h * 3.4)
        E = np.array(self.sky.sun_e) * SUN_GAIN
        self.sun_e = tuple(float(x) for x in E)
        omega = 6.8e-5
        self.sun_l = tuple(float(x) for x in np.minimum(E / omega, 3e4))
        self.hdr = gl.target(size)
        self.mist = gl.target((size[0] // 2, size[1] // 2))
        self.mist_src = gl.target((NS[0] // 2, NS[1] // 4), comps=1, mips=True)
        self.fract = self.sky._fractal()
        self._t_src = None
        self.blocks = gl.target((self.BLOCKS[0] * self.SUB, self.BLOCKS[1] * self.SUB), mips=True)
        self.blocks.tex.filter = (moderngl.NEAREST_MIPMAP_NEAREST, moderngl.NEAREST)
        self.down, s = [], (size[0] // 2, size[1] // 2)
        for i in range(6):
            self.down.append(gl.target(s))
            s = (max(s[0] // 2, 1), max(s[1] // 2, 1))
        self.up = [gl.target(d.size) for d in self.down[:-1]]

    def _bake_terrain(self):
        """the sand at the scale of a pixel: computed, not stored, in under a second."""
        gl, ctx = self.gl, self.gl.ctx
        texel = .0008
        rho0, rho1 = float(np.log(bs.R0)), float(np.log(140.0))
        th0, th1 = bs.TH0, bs.TH1
        W, H = int((th1 - th0) / texel), int((rho1 - rho0) / texel)
        self.ter = (th0, th1, rho0, rho1)
        self.t0 = ctx.texture((W, H), 4, dtype="f2")
        self.t1 = ctx.texture((W, H), 4, dtype="f2")
        fbo = ctx.framebuffer([self.t0, self.t1])
        fbo.use()
        prog = gl.prog("beach_terrain.frag")
        gl.bind(prog, dict(uTer=self.ter, uTexel=texel, uBed=self.sim.bedtex, uPol=self.sim.grid, uNS=self.sim.size))
        vao = ctx.simple_vertex_array(prog, gl.tri, "p")
        vao.render(moderngl.TRIANGLES, vertices=3)
        for t in (self.t0, self.t1):
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.anisotropy = 16.0
            t.repeat_x = t.repeat_y = False

    def _beach(self, t, ease):
        if self._t_view == (t, ease):
            return
        gl = self.gl
        self.sim.seek(t)
        self.rfbo.use()
        prog = gl.prog("beach_view.frag")
        gl.bind(prog, dict(uS=self.sim.A, uB=self.sim.B, uM=self.sim.M, uLaceN=self.lace_n, uGrid=self.sim.grid, uBlend=self.sim.mat_blend(t)))
        ctx = gl.ctx
        if "beach_view" not in gl._vaos:
            gl._vaos["beach_view"] = ctx.simple_vertex_array(prog, gl.tri, "p")
        gl._vaos["beach_view"].render(moderngl.TRIANGLES, vertices=3)
        self.r1.gen()
        r0 = self.r0
        if ease:
            gl.run("beach_soften.frag", self.r0s, uR0=self.r0, uGrid=self.sim.grid)
            r0 = self.r0s
        r0.gen()
        gl.run("beach_hmax.frag", self.hmax, uR0=r0, uBlock=self.block, uNS=self.sim.size)
        self._t_view = (t, ease)

    # ---------------------------------------------------------------- per-frame
    def _ocean(self, t):
        if self._t_ocean != t:
            A, B = self.ocean.at(t)
            self.tA.write(A.tobytes())
            self.tB.write(B.tobytes())
            self.tA.build_mipmaps()
            self.tB.build_mipmaps()
            self._t_ocean = t

    def _main_uniforms(self, st):
        t = st["t"]
        birds = self.flock.at(t)
        drift = []
        for (wx, wz), alt in zip(SKY["drift"], LAYER_ALT):
            k = .001 * t / alt
            drift.append((-wx * k, 0.0, -wz * k))
        u = dict(
            uAspect=self.aspect, uT=t, uH=CAM_H, uCurv=.5 / RE_EFF, uHmaxO=self.ocean_hmax, uCas=self.cas,
            uA=12, uB=11, uDome=13, uCloud=14, uSkyWin=(float(AZ_RANGE[0]), float(AZ_RANGE[1]), float(EL_RANGE[0]), float(EL_RANGE[1])), uBg=self.sky.bg, uEnv=self.sky.env, uSun=self.sky.sun_dir,
            uSunE=self.sun_e, uSunL=self.sun_l, uDrift=drift, uAir=AIR_BETA,
            uBird=[tuple(float(x) for x in b) for b in birds],
            uFigure=(FIGURE_AZ, FIGURE_EL, FIGURE_H, 0.0),
            uR0=self.r0, uR1=self.r1, uHmax=self.hmax, uMat=self.sim.M, uMatBlend=self.sim.mat_blend(t), uMatLag=self.sim.mat_lag(), uT0=self.t0, uT1=self.t1, uLace=self.lace, uLaceTh=self.lace_th, uLaceN=self.lace_n, uLaceL=TILE, uStacks=self.sky.stacks,
            uStackWin=(float(STACK_AZ[0]), float(STACK_AZ[1]), float(STACK_EL[0]), float(STACK_EL[1])),
            uPol=self.sim.grid, uNS=self.sim.size, uTer=self.ter, uRmax=bs.R1, uBlock=float(self.block), uTop=3.0,
        )
        return u

    def _bind_arrays(self):
        self.tA.use(location=12)
        self.tB.use(location=11)
        self.sky.dome.use(location=13)
        (self.clouds_h if self._hero_now else self.sky.clouds).use(location=14)

    def render(self, st: dict, out: Target):
        gl, ctx = self.gl, self.gl.ctx
        t = st["t"]
        self._ocean(t)
        yaw, pitch, roll = st.get("yaw", 0.0), st.get("pitch", -.02), st.get("roll", 0.0)
        h = float(np.clip((yaw - .45) / .3, 0.0, 1.0))
        hero = 1.0 - h * h * (3.0 - 2.0 * h)
        self._beach(t, hero > 0.0)
        self._hero_now = hero > 0.0
        self._bind_arrays()
        tan_half = st.get("tan_half", .27)
        u = self._main_uniforms(st)
        if hero > 0.0:
            u.update(uR0=self.r0s, uLace=self.lace_h, uLaceN=self.lace_nh, uLaceTh=self.lace_thh, uLaceL=TILE_READABLE, uStacks=self.stacks_h)
        drop = st.get("drop")
        crack = st.get("crack")
        u.update(uHero=hero, uMode=0, uTan=tan_half, uDrop=(0.0, 0.0, 1.0, 0.0), uCrack=(0.0, 0.0, 0.0, 0.0), uGain=1.0, uHorEl=-HORIZON_DIP, uJit=(0.0, 0.0), uFoot=1.0,
                 uTile=(float(self.TILE_AZ), float(self.TILE_EL)), uGrid=self.BLOCKS, uSub=self.SUB)
        if drop:
            u.update(uMode=1, uDrop=(drop["cx"], drop["cy"], drop["R"], drop["lens"]))
        # the world streaming in: coarse blocks first
        crack_on = bool(crack) and t < crack["t_arrive"] + 1.4 + .55 + 1e-4
        if crack_on:
            F, R, U = _basis(0.0, 0.0, 0.0)
            ub = dict(u, uF=F, uR=R, uU=U, uMode=2, uEq=(crack["edge_az"], -HORIZON_DIP))
            gl.run("sea_main.frag", self.blocks, **ub)
            self.blocks.gen()
            self.blocks.tex.use(location=15)
            u.update(uCrack=(1.0, crack["t_arrive"], crack["edge_az"], 0.0), uBlocks=15)
            self._bind_arrays()
        # shutter: a whip pan integrates several cameras over 1/50 s
        yaw_v = st.get("yaw_v", 0.0)
        smear = abs(yaw_v) / 50.0
        n = int(np.clip(np.ceil(smear / .0028), 1, 14)) if smear > .0009 else 1
        # in the shot itself a pixel is several looks: thin foam seen at a grazing angle has more in a pixel
        # than any one filtered lookup can say. The far view from the flower only needs the light.
        aa = len(_AA) if abs(tan_half - SHOT_TAN) < 1e-5 else 1
        looks = max(n, aa)
        self.hdr.fbo.use()
        blur_px = 0.0
        if looks == 1:
            F, R, U = _basis(yaw, pitch, roll)
            gl.run("sea_main.frag", self.hdr, uF=F, uR=R, uU=U, **u)
        else:
            self.hdr.clear()
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            px = 2.0 / self.size[1]
            for i in range(looks):
                dy = yaw_v * ((i + .5) / looks - .5) / 50.0 if n > 1 else 0.0
                F, R, U = _basis(yaw + dy, pitch, roll)
                j = _AA[i % len(_AA)] if aa > 1 else (0.0, 0.0)
                gl.run("sea_main.frag", self.hdr, uF=F, uR=R, uU=U, **dict(u, uGain=1.0 / looks, uJit=(j[0] * px, j[1] * px), uFoot=.6 if aa > 1 else 1.0))
            ctx.disable(moderngl.BLEND)
            blur_px = smear / looks / (tan_half / (self.size[1] / 2))
        if hero > 0.0:
            # the air is one exposure of one camera: in a whip it would stand still in a smeared world
            amount = hero * (1.0 - float(np.clip((abs(st.get("yaw_v", 0.0)) - .5) / 2.5, 0, 1)))
            if amount > 0.0:
                self._air(u, yaw, pitch, roll, tan_half, amount)
        # lens glare: a small fraction of every bright thing leaks into its neighbourhood
        prev = self.hdr
        for i, d in enumerate(self.down):
            gl.run("sea_down.frag", d, uTex=prev, uPx=(1 / prev.size[0], 1 / prev.size[1]), uFirst=1.0 if i == 0 else 0.0)
            prev = d
        for i in range(len(self.up) - 1, -1, -1):
            src = self.down[i + 1] if i == len(self.up) - 1 else self.up[i + 1]
            gl.run("sea_up.frag", self.up[i], uTex=src, uBase=self.down[i], uPx=(1 / src.size[0], 1 / src.size[1]),
                   uMix=(.22, .30, .36, .40, .42)[i])
        gl.run("sea_comp.frag", out, uHdr=self.hdr, uGlare=self.up[0], uTight=self.down[0],
               uPx=(1 / self.size[0], 1 / self.size[1]), uBlur=blur_px)

    def _air(self, u, yaw, pitch, roll, tan_half, amount):
        """the salt in the air, over the picture and before the lens: it glows, the lens must see it."""
        gl, ctx = self.gl, self.gl.ctx
        F, R, U = _basis(yaw, pitch, roll)
        self.fract.use(location=10)
        if self._t_src != u["uT"]:
            gl.run("beach_mistsrc.frag", self.mist_src, uR1=self.r1, uSpan=float(np.log(2600.0 / 16.0) / 40 / (bs.D * self.sim.size[1])))
            self.mist_src.gen()
            self._t_src = u["uT"]
        gl.run("beach_mist.frag", self.mist, uF=F, uR=R, uU=U, uAspect=self.aspect, uTan=tan_half, uT=u["uT"], uH=CAM_H, uSun=u["uSun"], uSunE=u["uSunE"],
               uDrift=u["uDrift"], uSkyWin=u["uSkyWin"], uPol=u["uPol"], uNS=u["uNS"], uSrc=self.mist_src, uEnv=u["uEnv"], uCloud=14, uFract=10,
               uFw=2.0 * tan_half / self.size[1], uAmount=amount,
               uStack=[(s[0], s[1], s[2] * max(s[4], 1.0), s[6]) for s in _STACKS])
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.SRC_ALPHA
        gl.run("beach_over.frag", self.hdr, uTex=self.mist)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.BLEND)

    # ---------------------------------------------------------------- lookdev camera
    def lookdev(self, t: float) -> dict:
        st = dict(t=t, yaw=0.0, pitch=-.095, roll=0.0, tan_half=.27)
        if t < 16.0:
            u = np.clip((t - 12.0) / 4.0, 0, 1)
            e = u * u * (3 - 2 * u)
            st["drop"] = dict(cx=float(.42 * (1 - e)), cy=float(-.16 * (1 - e)), R=float(.066 * (3.09 / .066) ** e), lens=float(1 - e))
            st["yaw"], st["pitch"] = 0.0, -.095
        elif t < 22.0:
            st["yaw"], st["pitch"] = float(_YAW(t)), float(_PITCH(t))
        elif t < 24.0:
            y0 = float(_YAW(22.0))
            yaw = y0 * (1 - min((t - 22) / .5, 1)) + float(_WHIP(t))
            st["yaw"], st["pitch"] = yaw, float(_PITCH(22.0))
            d = 1 / 240
            y1 = y0 * (1 - min((t + d - 22) / .5, 1)) + float(_WHIP(t + d))
            st["yaw_v"] = float((y1 - yaw) / d)
            st["crack"] = dict(t_arrive=22.25, edge_az=.42)
        else:
            st["yaw"], st["pitch"] = float(_YAW_LATE(t)), float(_PITCH(22.0))
            st["crack"] = dict(t_arrive=22.25, edge_az=.42)
        ev = -1.05
        isp = dict(ev=ev, iso=1.0, zebra=.96, sat=1.0)
        hud = dict(shutter="1/250", fnum="F8", iso="100", ev=f"{ev + .3:+.1f}", ev_meter=ev + .3 + .06 * np.sin(t * .7),
                   wb="AWB", look="ST", lens="35mm")
        ph = dict(defocus=.55, band=.012, ae_target=.13)
        return dict(world=st, isp=isp, hud=hud, ph=ph)
