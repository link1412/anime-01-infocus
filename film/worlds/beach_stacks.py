"""Sea stacks and the headland, baked once into a strip of the panorama: they never change, only
the air around them does, and at this distance it hardly does."""
from __future__ import annotations
import hashlib, pathlib, time
import numpy as np
import moderngl
from ..core import glsl
from . import beach_sim as bs

CACHE = pathlib.Path(__file__).resolve().parents[2] / "out" / "cache"
RE_EFF = 7.4e6                # effective earth radius including refraction
AZ = np.radians((-36.0, 86.0))
EL = np.radians((-.42, 6.2))
RES = np.radians(.025)
SIZE = (int(round((AZ[1] - AZ[0]) / RES)), int(round((EL[1] - EL[0]) / RES)))


def _polar(az, r):
    return float(r * np.sin(az)), float(r * np.cos(az))


# azimuth, range, plan radius, height, elongation, plan angle, seed, column width
_STACKS = [(-.243, 1150.0, 25.0, 58.0, 1.4, .5, 17.0, .85),
           (-.281, 1420.0, 13.0, 21.0, 1.9, -.3, 41.0, 1.0),
           (-.128, 1720.0, 9.0, 34.0, .9, .1, 73.0, 1.2),
           (.214, 960.0, 11.0, 13.0, 1.5, .9, 29.0, .75)]


def bake_stacks(gl, bg, sun_dir, sun_e, sky_level, frag="beach_stacks.frag"):
    key = hashlib.sha1((glsl(frag) + repr(_STACKS) + repr(SIZE) + repr(bs.CAM_H) + repr(sky_level) + repr(tuple(sun_e)) + repr(tuple(sun_dir))).encode()).hexdigest()[:12]
    path = CACHE / f"beach_stacks_{key}.npy"
    if path.exists():
        data = np.load(path)
    else:
        t0 = time.time()
        t = gl.target(SIZE)
        S, S2 = [], []
        for az, r, rad, h, el, ang, seed, cs in _STACKS:
            x, z = _polar(az, r)
            S.append((x, z, rad, h))
            S2.append((el, ang, seed, cs))
        S.append((0.0, 0.0, 0.0, 200.0))
        S2.append((1.0, 0.0, 91.0, 3.0))
        t.fbo.use()
        t.clear()
        ctx = gl.ctx
        tile = 16
        for y in range(0, SIZE[1], tile):
            for x in range(0, SIZE[0], tile):
                t.fbo.scissor = (x, y, tile, tile)
                gl.run(frag, t, uRange=(float(AZ[0]), float(AZ[1]), float(EL[0]), float(EL[1])), uH=bs.CAM_H, uCurv=.5 / RE_EFF,
                       uPix=float(RES), uSun=sun_dir, uSunE=sun_e, uSkyLevel=float(sky_level), uBg=bg, uS=S, uS2=S2)
                ctx.finish()
        t.fbo.scissor = None
        data = np.frombuffer(t.fbo.read(components=4, dtype="f2"), "f2").reshape(SIZE[1], SIZE[0], 4).copy()
        t.tex.release()
        t.fbo.release()
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
        print(f"stacks: baked in {time.time() - t0:.1f}s", flush=True)
    tex = gl.ctx.texture(SIZE, 4, data.tobytes(), dtype="f2")
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.anisotropy = 16.0
    tex.repeat_x = tex.repeat_y = False
    return tex
