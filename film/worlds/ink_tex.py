"""Materials baked once: brush noise, xuan paper, fibre conductivity, mist, glyph atlas."""
from __future__ import annotations
import numpy as np
import moderngl
from PIL import Image, ImageDraw
from .. import fonts
from .ink_brush import Marks, FMT, NAMES, bez, bell

PAPER_P = 4.0      # view units per paper tile


def uniformize(a, q=2048):
    """Flatten the marginal to uniform [0,1] so thresholds mean 'fraction of area'."""
    lo = np.quantile(a[::3, ::3], np.linspace(0, 1, q))
    lo = np.maximum.accumulate(lo + np.arange(q) * 1e-9)
    return np.interp(a, lo, np.linspace(0, 1, q)).astype("f4")


def fft_noise(rng, n, m=None, beta=2.0, ax=1.0, ay=1.0, lo=0.0, hi=None):
    """Tileable noise. beta: spectral slope. lo/hi: band limits in cycles per tile (after anisotropy)."""
    m = m or n
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.rfftfreq(m)[None, :] * m
    r = np.sqrt((fx * ax) ** 2 + (fy * ay) ** 2)
    r[0, 0] = 1
    amp = r ** (-beta / 2)
    if lo > 0:
        amp *= 1 - np.exp(-(r / lo) ** 4)
    if hi:
        amp *= np.exp(-(r / hi) ** 4)
    amp[0, 0] = 0
    w = rng.normal(0, 1, (n, m))
    return np.fft.irfft2(np.fft.rfft2(w) * amp, s=(n, m)).astype("f4")


def brush_noise(seed=11):
    rng = np.random.default_rng(seed)
    n = 1024
    return np.stack([
        uniformize(fft_noise(rng, n, beta=2.6, lo=2.0)),
        uniformize(fft_noise(rng, n, beta=.8, lo=30, hi=110)),
        uniformize(fft_noise(rng, n, beta=.5, lo=170, hi=420)),
        uniformize(fft_noise(rng, n, beta=.5, lo=70, hi=190)),
    ], -1)


def draw_marks(gl, target, m: Marks, canvas, origin=(0.0, 0.0), offs=((0.0, 0.0),), noise=None, tooth=.66, blend=None):
    """Rasterise an ordered mark list into a target, with wrap copies."""
    ctx = gl.ctx
    arr = m.array()
    if len(arr) == 0:
        return
    prog = gl.prog("ink_mark.frag", "ink_mark.vert")
    vbo = ctx.buffer(arr.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, FMT, *NAMES)], skip_errors=True)
    gl.bind(prog, dict(uNoise=noise, uTooth=tooth, uCanvas=canvas, uOrigin=origin))
    target.fbo.use()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = blend or (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
    for off in offs:
        prog["uOff"].value = off
        vao.render(moderngl.TRIANGLE_STRIP)
    ctx.disable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    vao.release()
    vbo.release()


def tex2d(gl, arr, mips=True, repeat=True, comps=None):
    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    comps = comps or (arr.shape[2] if arr.ndim == 3 else 1)
    t = gl.ctx.texture((w, h), comps, arr.tobytes(), dtype="f1" if arr.dtype == np.uint8 else "f4")
    t.repeat_x = t.repeat_y = repeat
    if mips:
        t.build_mipmaps()
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    else:
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return t


def fibres(rng, n, lmin, lmax, wmin, wmax, P):
    """A sheet of long fibres scattered with random orientation, in [0,P)^2."""
    m = Marks()
    M = 14
    c = rng.uniform(0, P, (n, 2))
    th = rng.uniform(0, np.pi, n)
    ln = np.exp(rng.uniform(np.log(lmin), np.log(lmax), n))
    d = np.stack([np.cos(th), np.sin(th)], 1) * ln[:, None] / 2
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    p0, p2 = c - d, c + d
    p1 = c + nrm * rng.normal(0, .55, (n, 1))
    paths = bez(p0, p1, p2, M)
    u = np.linspace(0, 1, M)[None]
    hw = (np.exp(rng.uniform(np.log(wmin), np.log(wmax), n))[:, None]) * (bell(u, .25))
    m.strokes(paths, hw, load=rng.uniform(.45, 1.0, n)[:, None] * np.ones((n, M)), dry=0.0, edge=.2, seed=rng.random(n))
    return m


def bake_paper(gl, noise_tex, seed=23):
    """Returns (paper RGBA8 4096^2: r=tone, g=embossed fibre relief, b=tooth, a=visible fibre), (K R8 2048^2)."""
    rng = np.random.default_rng(seed)
    n = 4096
    P = PAPER_P
    offs = [(dx, dy) for dx in (-P, 0.0, P) for dy in (-P, 0.0, P)]

    def sheet(size, marks):
        t = gl.target((size, size), comps=1, dtype="f2")
        t.clear()
        draw_marks(gl, t, marks, (P, P), offs=offs, noise=noise_tex,
                   blend=(moderngl.ONE, moderngl.ONE))
        a = np.frombuffer(t.tex.read(), "f2").reshape(size, size).astype("f4")
        t.fbo.release(); t.tex.release()
        return a

    # ---- what the eye sees: a mat of fine fibres plus rarer long, coarse ones
    fine = sheet(n, fibres(rng, 62000, .018, .16, .00042, .0009, P))
    coarse = sheet(n, fibres(rng, 2000, .06, .30, .0009, .0019, P))
    fib = np.clip(fine * .55 + coarse * .7, 0, 1.0)
    floc = uniformize(fft_noise(rng, n, beta=1.4, lo=30, hi=240))
    cloud = uniformize(fft_noise(rng, n, beta=2.4, lo=1.5))
    grain = uniformize(fft_noise(rng, n, beta=.6, lo=500, hi=1500))
    tone = .5 + (floc - .5) * .42 + (cloud - .5) * .30 + (grain - .5) * .14
    gy, gx = np.gradient(np.clip(fine * .6 + coarse, 0, 1))
    relief = .5 + np.clip((gx + gy) * 1.2, -.5, .5)
    tooth = uniformize(fft_noise(rng, n, beta=.8, lo=120, hi=700))
    paper = np.stack([tone, relief, tooth, fib], -1)
    paper = (np.clip(paper, 0, 1) * 255 + .5).astype("u1")

    # ---- what the water sees: conductive strands that percolate. Wider than visible fibres,
    # one texel here is about one world pixel at zoom 1.
    nk = 2048
    strands = sheet(nk, fibres(rng, 3300, .05, .50, .0016, .0034, P))
    strands2 = sheet(nk, fibres(rng, 520, .25, 1.3, .0020, .0040, P))
    c2 = uniformize(fft_noise(rng, nk, beta=2.2, lo=2.0))
    s = np.clip(strands * 1.2 + strands2 * 1.2, 0, 1)
    K = (.030 + .970 * s ** 1.3) * (.10 + 1.25 * c2 ** 1.2)
    K = np.clip(K, 0, 1)
    K = (K * 255 + .5).astype("u1")
    return paper, K


def bake_mist(seed=31):
    """Two-channel mist tile (r: long drifting bands, g: soft curls), elongated horizontally."""
    rng = np.random.default_rng(seed)
    h, w = 512, 2048
    a = uniformize(fft_noise(rng, h, w, beta=2.9, ax=2.8, ay=1.0, lo=1.5))
    b = uniformize(fft_noise(rng, h, w, beta=2.4, ax=1.6, ay=1.0, lo=3.0))
    return (np.stack([a, b], -1) * 255 + .5).astype("u1")


GLYPHS = [("山", "Kaiti.ttc", 0), ("山", "Songti.ttc", 6), ("山", "Xingkai.ttc", 0),
          ("水", "Kaiti.ttc", 0), ("水", "Songti.ttc", 6), ("水", "Xingkai.ttc", 0),
          ("云", "Kaiti.ttc", 0), ("云", "Songti.ttc", 6), ("云", "Xingkai.ttc", 0),
          ("日", "Kaiti.ttc", 0), ("日", "Songti.ttc", 6)]
ATLAS_RES = 256


def bake_atlas():
    """Glyphs as one layer each, so every mip stays inside its glyph and the top mip is its mean coverage."""
    R = ATLAS_RES
    S = 4
    out = np.zeros((len(GLYPHS), R, R), "u1")
    means = []
    for i, (ch, fn, idx) in enumerate(GLYPHS):
        f = fonts.font(fn, int(R * S * .86), idx)
        im = Image.new("L", (R * S, R * S), 0)
        d = ImageDraw.Draw(im)
        l, t, r, b = d.textbbox((0, 0), ch, font=f)
        d.text(((R * S - (r - l)) / 2 - l, (R * S - (b - t)) / 2 - t), ch, font=f, fill=255)
        a = np.asarray(im.resize((R, R), Image.LANCZOS), "f4") / 255
        out[i] = (np.flipud(a) * 255 + .5).astype("u1")
        means.append(float(a.mean()))
    return out, means
