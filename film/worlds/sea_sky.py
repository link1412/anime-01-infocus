"""The sky of the beach: an atmosphere and three cloud layers raymarched once (cached), then only drifted.

Two bakes of one sky: the whole dome at low resolution (it is what the water and the flower see
all around), and the window the camera can actually look at, at about .018 deg per pixel, one to a pixel of the frame.
"""
from __future__ import annotations
import hashlib, pathlib, time
import numpy as np
import moderngl
from scipy.ndimage import binary_dilation, gaussian_filter, label
from ..core import GL, glsl
from .beach_stacks import bake_stacks, AZ as STACK_AZ, EL as STACK_EL

CACHE = pathlib.Path(__file__).resolve().parents[2] / "out" / "cache"

SUN_AZ = 0.0
SUN_EL = np.radians(3.5)

BG = (2048, 768)              # clear sky, whole dome, v = sqrt(el)
DOME = (2048, 640)            # clouds, whole dome, v = sqrt(el)
ENV = (2048, 512)             # what the water reflects, v linear
AZ_RANGE = np.radians((-36.0, 86.0))
EL_RANGE = np.radians((-.3, 22.2))
RES = np.radians(.018)
WINDOW = (int(round((AZ_RANGE[1] - AZ_RANGE[0]) / RES)), int(round((EL_RANGE[1] - EL_RANGE[0]) / RES)))
SKY = dict(
    sun_el=float(SUN_EL), haze=.03, e0=4.2, wb=(.873, 1.0, .93), fill=.12, cold=.5, sun_tint=(1.0, .86, .66),
    layers=[
        dict(shells=(.62, 1.25), sigma=4.0, amb=.3, sun=.5, steps=256, light_el=.012, off=(137.0, 58.0)),
        dict(shells=(2.1, 3.5), sigma=9.0, amb=.2, sun=.5, steps=1600, light_el=-.012, off=(-31.0, 212.0)),
        dict(shells=(7.4, 8.2), sigma=.03, amb=.45, sun=.7, steps=1, light_el=-.03, off=(411.0, -77.0)),
    ],
    # m/s of drift of each layer, world x,z
    drift=[(5.2, -1.2), (7.5, 2.0), (12.0, 5.0)],
)
LAYER_ALT = [.9, 2.8, 7.8]
SCRAP_OPACITY = .16           # what counts as cloud when looking for ones standing alone
SCRAP_AREA = 45000            # and how big one has to be to stay, in window pixels


def _make_noise():
    """Tileable Perlin-Worley volumes for the clouds, cached."""
    path = CACHE / "sea_noise_v2.npz"
    if path.exists():
        d = np.load(path)
        return d["a"], d["b"]
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(2024)

    def worley(N, freq):
        pts = (np.stack(np.meshgrid(*[np.arange(freq)] * 3, indexing="ij"), -1).reshape(-1, 3) + rng.random((freq ** 3, 3))) / freq
        g = (np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1).reshape(-1, 3) + .5) / N
        d, _ = cKDTree(pts, boxsize=1.0).query(g, workers=-1)
        return (1 - np.clip(d * freq, 0, 1)).reshape(N, N, N)

    def wfbm(N, f):
        return .625 * worley(N, f) + .25 * worley(N, f * 2) + .125 * worley(N, f * 4)

    def spectral(N, beta, lo):
        w = rng.normal(size=(N, N, N))
        F = np.fft.fftn(w)
        n = np.fft.fftfreq(N, 1 / N)
        k = np.sqrt(sum(a ** 2 for a in np.meshgrid(n, n, n, indexing="ij")))
        F *= np.where(k >= lo, np.maximum(k, 1) ** -beta, 0)
        s = np.fft.ifftn(F).real
        return s / s.std()

    N = 128
    per = np.clip(.5 + .24 * spectral(N, 1.45, 2), 0, 1)
    w4 = wfbm(N, 4)
    pw = np.clip(per + (w4 - .5) * .38 * (per * .5 + .5), 0, 1)   # Perlin, gently billowed by Worley
    a = np.stack([pw, wfbm(N, 5), wfbm(N, 9), wfbm(N, 17)], -1)
    b = np.stack([wfbm(64, 4), wfbm(64, 7), wfbm(64, 12)], -1)
    a = (a * 255).astype("u1")
    b = (b * 255).astype("u1")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, a=a, b=b)
    return a, b


def _make_fractal():
    """A plain tileable fractal volume: one lookup brings every octave down to the voxel."""
    path = CACHE / "beach_fractal_v1.npy"
    if path.exists():
        return np.load(path)
    rng = np.random.default_rng(90210)
    N = 256
    n = np.fft.fftfreq(N, 1 / N).astype("f4")
    k2 = n[:, None, None] ** 2 + n[None, :, None] ** 2 + n[None, None, :N // 2 + 1] ** 2
    F = np.fft.rfftn(rng.normal(size=(N, N, N)).astype("f4"))
    F *= np.where(k2 > 0, np.maximum(k2, 1) ** (-.5 * 1.42), 0)
    s = np.fft.irfftn(F, s=(N, N, N)).real
    s = np.clip(.5 + .2 * s / s.std(), 0, 1)
    v = (s * 255).astype("u1")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, v)
    return v


def _key():
    src = "".join(glsl(n) for n in ("sea_bake_atmo.frag", "sea_bake_cloud.frag"))
    return hashlib.sha1((src + repr(SKY) + repr(BG) + repr(DOME) + repr(WINDOW) + "f1").encode()).hexdigest()[:12]


class SkyBake:
    """Bakes (or loads) clear sky, cloud panoramas and the reflection environment."""

    def __init__(self, gl: GL):
        self.gl = gl
        ctx = gl.ctx
        key = _key()
        path = CACHE / f"beach_sky_{key}.npz"
        sd = (np.cos(SUN_EL) * np.sin(SUN_AZ), np.sin(SUN_EL), np.cos(SUN_EL) * np.cos(SUN_AZ))
        self.sun_dir = tuple(float(x) for x in sd)
        self.base = dict(uSunDir=self.sun_dir, uHaze=SKY["haze"], uE0=SKY["e0"], uWB=SKY["wb"], uFill=SKY["fill"], uCold=SKY["cold"])
        if path.exists():
            d = np.load(path)
            bg, dome, window, self.sun_e = d["bg"], d["dome"], d["window"], tuple(float(x) for x in d["sun_e"])
        else:
            t0 = time.time()
            bg, dome, window = self._bake()
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(path, bg=bg, dome=dome, window=window, sun_e=np.array(self.sun_e))
            print(f"sky: baked in {time.time() - t0:.1f}s -> {path.name}", flush=True)
        n = len(SKY["layers"])
        self.bg = ctx.texture(BG, 4, bg.tobytes(), dtype="f2")
        self.bg.repeat_x, self.bg.repeat_y = True, False
        self.bg.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.dome = ctx.texture_array((DOME[0], DOME[1], n), 4, dome.tobytes(), dtype="f2")
        self.dome.repeat_x, self.dome.repeat_y = True, False
        self.dome.build_mipmaps()
        self.dome.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.dome.anisotropy = 16.0
        self.clouds = ctx.texture_array((WINDOW[0], WINDOW[1], n), 4, window.tobytes(), dtype="f2")
        self.clouds.repeat_x, self.clouds.repeat_y = False, False
        self.clouds.build_mipmaps()
        self.clouds.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.clouds.anisotropy = 16.0
        self.stacks = bake_stacks(gl, self.bg, self.sun_dir, tuple(self.sun_e), .5)
        self.env = gl.target(ENV, mips=True, clamp=False)
        self.env.tex.repeat_y = False
        self.dome.use(location=14)
        gl.run("sea_bake_env.frag", self.env, uBg=self.bg, uCloud=14, uStacks=self.stacks,
               uStackWin=(float(STACK_AZ[0]), float(STACK_AZ[1]), float(STACK_EL[0]), float(STACK_EL[1])))
        self.env.gen()

    def _fractal(self):
        nf = self.gl.ctx.texture3d((256, 256, 256), 1, _make_fractal().tobytes(), dtype="f1")
        nf.repeat_x = nf.repeat_y = nf.repeat_z = True
        nf.build_mipmaps()
        nf.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        return nf

    def _bake_layers(self, tgt, size, rng, pix, step_scale):
        gl, ctx = self.gl, self.gl.ctx
        out = np.zeros((len(SKY["layers"]), size[1], size[0], 4), "f2")
        for i, ly in enumerate(SKY["layers"]):
            le = ly["light_el"]
            light = (0.0, np.sin(SUN_EL + le), np.cos(SUN_EL + le))
            tgt.fbo.use()
            tgt.clear()
            tile = 512
            for y in range(0, size[1], tile):
                for x in range(0, size[0], tile):
                    tgt.fbo.scissor = (x, y, tile, tile)
                    gl.run("sea_bake_cloud.frag", tgt, uNA=12, uNB=13, uNF=11, uLayer=i, uLightDir=tuple(float(v) for v in light),
                           uShells=ly["shells"], uOff=ly["off"], uSigma=ly["sigma"], uAmb=ly["amb"], uSunK=ly["sun"],
                           uSteps=max(int(ly["steps"] * step_scale), 1), uRange=rng, uPix=pix, **self.base)
                    ctx.finish()
            tgt.fbo.scissor = None
            out[i] = np.frombuffer(tgt.fbo.read(components=4, dtype="f2"), "f2").reshape(size[1], size[0], 4)
        return out

    def hero_clouds(self):
        """The window of clouds for the views facing the sun, the small ones standing alone taken out: in that much empty
        sky they read as scraps cut out and stuck on. A copy, so the dome, the reflections and the side view keep theirs."""
        window = np.load(CACHE / f"beach_sky_{_key()}.npz")["window"]
        path = CACHE / f"beach_cloudkeep2_{_key()}_{SCRAP_OPACITY}_{SCRAP_AREA}.npz"
        if path.exists():
            drop = np.load(path)["drop"]
        else:
            drop = np.zeros(window.shape[:3], "u1")
            # (the top layer is a thin veil over everything: a thick patch of it is no cloud alone, and it would leave a hole)
            for i in range(len(window) - 1):
                lab, n = label(window[i, :, :, 3] < 1.0 - SCRAP_OPACITY)
                area = np.bincount(lab.ravel(), minlength=n + 1)
                small = area < SCRAP_AREA
                small[0] = False
                drop[i] = binary_dilation(small[lab], iterations=5)
            np.savez_compressed(path, drop=drop)
        for i in range(len(window)):
            m = gaussian_filter(drop[i].astype("f4"), 3.0)[..., None]
            m = np.clip(m * 1.6, 0.0, 1.0)
            window[i] = window[i] * (1.0 - m) + np.array((0.0, 0.0, 0.0, 1.0), "f4") * m
        tex = self.gl.ctx.texture_array((WINDOW[0], WINDOW[1], len(window)), 4, window.astype("f2").tobytes(), dtype="f2")
        tex.repeat_x, tex.repeat_y = False, False
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 16.0
        return tex

    def _bake(self):
        gl, ctx = self.gl, self.gl.ctx
        one = gl.target((1, 1), dtype="f4")
        gl.run("sea_bake_atmo.frag", one, uMode=1, uAlt=.02, **self.base)
        e = np.frombuffer(one.fbo.read(components=4, dtype="f4"), "f4")[:3]
        l = float(np.dot(e, (.25, .65, .10)))
        self.sun_e = tuple(float(x) for x in np.array(SKY["sun_tint"]) * l)   # the same cold camera
        t_bg = gl.target(BG)
        gl.run("sea_bake_atmo.frag", t_bg, uMode=0, uAlt=.02, **self.base)
        bg = np.frombuffer(t_bg.fbo.read(components=4, dtype="f2"), "f2").reshape(BG[1], BG[0], 4).copy()
        a, b = _make_noise()
        na = ctx.texture3d((128, 128, 128), 4, a.tobytes(), dtype="f1")
        nb = ctx.texture3d((64, 64, 64), 3, b.tobytes(), dtype="f1")
        for n in (na, nb):
            n.repeat_x = n.repeat_y = n.repeat_z = True
            n.build_mipmaps()
            n.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        na.use(location=12)
        nb.use(location=13)
        self._fractal().use(location=11)
        dome = self._bake_layers(gl.target(DOME), DOME, (0.0, 0.0, 0.0, 0.0), .004, .4)
        t = time.time()
        window = self._bake_layers(gl.target(WINDOW), WINDOW, (AZ_RANGE[0], AZ_RANGE[1], EL_RANGE[0], EL_RANGE[1]), RES, 1.0)
        # each pixel's march starts at its own random offset: left in, that noise beats against the frame's own grid
        for i in range(len(window)):
            window[i] = gaussian_filter(window[i].astype("f4"), (.85, .5, 0))
        print(f"sky: window {WINDOW} in {time.time() - t:.1f}s", flush=True)
        return bg, dome, window
