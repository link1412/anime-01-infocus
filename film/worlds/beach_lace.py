"""The foam's own structure, computed when the world starts: coverage at four amounts of foam, so that
a mip of it is the true coverage of a pixel however far away and however stretched by perspective."""
from __future__ import annotations
import numpy as np
import moderngl

SIZE = 2048
TILE = 8.0                        # metres; beach_surface.glsl knows this
TILE_READABLE = 16.0              # metres of the field the view facing the sun uses; passed as uLaceL
LEVELS = (.07, .22, .50, .82)     # share of the area covered at each channel; beach_surface.glsl knows these


def _unit(a):
    lo, hi = np.quantile(a, [.01, .99])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def bake_lace(gl, readable=False):
    t = gl.target((SIZE, SIZE), dtype="f4")
    gl.run("beach_lace.frag", t, uReadable=1.0 if readable else 0.0)
    d = np.frombuffer(t.fbo.read(components=4, dtype="f4"), "f4").reshape(SIZE, SIZE, 4).astype(np.float64)
    t.tex.release()
    t.fbo.release()
    phi = d[..., 2]
    cover = np.empty((SIZE, SIZE, 4), "f2")
    ths = np.quantile(phi, [1 - a for a in LEVELS])
    for i, th in enumerate(ths):
        s = np.clip((phi - th) / .03 + .5, 0, 1)
        cover[..., i] = s * s * (3 - 2 * s)
    lo, hi = np.quantile(phi, [.02, .98])
    # r, g: where the rafts lie, two sizes. b: the field itself. a: the speckle of single bubbles
    relief = np.stack([_unit(d[..., 0]), _unit(d[..., 1]), np.clip((phi - lo) / (hi - lo), 0, 1), _unit(d[..., 3])], -1).astype("f2")
    out = []
    for arr in (cover, relief):
        tex = gl.ctx.texture((SIZE, SIZE), 4, arr.tobytes(), dtype="f2")
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 16.0
        tex.repeat_x = tex.repeat_y = True
        out.append(tex)
    # the same four amounts as heights of the field in the relief's blue, for looking at it up close
    return out[0], out[1], tuple(float(x) for x in (ths - lo) / (hi - lo))
