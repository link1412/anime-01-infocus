"""Whoever is holding the camera.

The film never sees them. What the glass holds is someone standing there, and the camera
in their hands, and the focus never gets as far as either. The camera is the only solid thing.
"""
from __future__ import annotations
import hashlib, pathlib
import numpy as np
import moderngl
from PIL import Image, ImageDraw
from ..core import GL, Target, glsl, smooth
from .. import fonts

EYE = (.130, .455)          # where the box ends up
FOREARM_L = ((-1.04, -1.18), (-.40, -.03))
FOREARM_R = ((.24, -.12), (1.05, -1.18))
GAP = .034
FORM = .013                 # how bright they are, in the glass: the little a terminal lights, on someone in a dark room
COOL = (.40, .47, .60)


def _engraving():
    """what's cut into the front of a lens: its focal length and speed, and nothing that belongs to anybody."""
    W, H = 2048, 64
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    f = fonts.font("Helvetica.ttc", 50, 0)
    for x, text in ((140, "55mm  1:1.8"), (1120, "\u2300 49mm")):
        d.text((x, H // 2), text, font=f, fill=255, anchor="lm")
    return np.flipud(np.asarray(im, "u1"))


def camera_plate(gl: GL, size, cam, spp=4096, cache=True, scale=1.6):
    """the camera in the glass, path traced: float rgb + coverage, rows bottom first. takes minutes once."""
    W, H = size
    key = hashlib.sha1(repr((glsl("desk_cam_pt.frag"), tuple(size), sorted(cam.items()), spp, scale)).encode()).hexdigest()[:12]
    path = pathlib.Path("out/cache") / f"camera_{key}.npy"
    if cache and path.exists():
        return np.load(path)
    eng = gl.ctx.texture((2048, 64), 1, _engraving().tobytes(), dtype="f1")
    eng.build_mipmaps()
    eng.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    eng.repeat_x = True
    acc = gl.target(size, dtype="f4", nearest=True)
    acc.clear()
    ctx = gl.ctx
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.ONE, moderngl.ONE
    for i in range(spp):
        gl.run("desk_cam_pt.frag", acc, uRes=(float(W), float(H)), uScale=scale, uBatch=float(i), uEngrave=eng, **cam)
        if i % 8 == 7:
            ctx.finish()
    ctx.disable(moderngl.BLEND)
    a = np.frombuffer(acc.fbo.read(components=4, dtype="f4"), "f4").reshape(H, W, 4).copy()
    cov = a[..., 3:]
    a[..., :3] = a[..., :3] / np.maximum(cov, 1e-3)
    a[..., 3] = cov[..., 0] / spp
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, a.astype("f4"))
    return a.astype("f4")


def _smin(a, b, k):
    h = np.clip(.5 + .5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _ell(x, y, c, rx, ry):
    return (np.sqrt(((x - c[0]) / rx) ** 2 + ((y - c[1]) / ry) ** 2) - 1) * min(rx, ry)


def _cap(x, y, a, b, r):
    a, b = np.array(a, float), np.array(b, float)
    ba = b - a
    h = np.clip(((x - a[0]) * ba[0] + (y - a[1]) * ba[1]) / (ba @ ba), 0, 1)
    return np.hypot(x - a[0] - ba[0] * h, y - a[1] - ba[1] * h) - r


def _taper(x, y, a, b, ra, rb):
    a, b = np.array(a, float), np.array(b, float)
    ba = b - a
    h = np.clip(((x - a[0]) * ba[0] + (y - a[1]) * ba[1]) / (ba @ ba), 0, 1)
    return np.hypot(x - a[0] - ba[0] * h, y - a[1] - ba[1] * h) - (ra + (rb - ra) * h)


def silhouette(x, y):
    """(mask, shade, forearms). view units, y up. Someone standing there, nothing more: hair, shoulders, two arms up to the camera."""
    cx = .015
    # head turned a little, so that it isn't a road sign
    c, s = np.cos(.055), np.sin(.055)
    hx = c * (x - cx) - s * (y - .20)
    hy = s * (x - cx) + c * (y - .20)
    head = _smin(_ell(hx, hy, (0, .32), .284, .322), _ell(hx, hy, (0, .15), .228, .262), .10)
    head = _smin(head, _ell(hx, hy, (-.012, .385), .308, .262), .03)       # hair
    for sg in (-1, 1):
        head = _smin(head, _ell(hx, hy, (sg * .292, .185), .034, .068), .012)
    neck = _cap(x, y, (cx, .25), (cx, -.30), .172)
    shoulders = _cap(x, y, (-.60, -.50), (.64, -.50), .27)
    trunk = _smin(neck, shoulders, .20)
    trunk = np.minimum(trunk, np.maximum(np.abs(x - cx) - .885, y + .50))
    trunk = np.minimum(trunk, np.minimum(_cap(x, y, (-.75, -.55), (-1.0, -1.3), .135), _cap(x, y, (.79, -.55), (1.03, -1.3), .135)))
    arms = np.minimum(_taper(x, y, *FOREARM_L, .128, .09), _taper(x, y, *FOREARM_R, .09, .128))
    edge = lambda sd: np.clip(.5 - sd * 477 / 1.5, 0, 1)
    solid = edge(np.minimum(np.minimum(trunk, arms), head))
    # type stops short of the arms, or they melt into the chest
    mask = edge(np.minimum(np.minimum(np.maximum(trunk, -(arms - GAP)), arms), head))
    # only a form, lit from below by what it is holding
    dx, dy = (x - cx) / .30, (y - .45) / .42
    nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
    sh_head = .50 + .50 * np.clip(dx * -.20 + dy * -.42 + nz * .86, 0, 1)
    sh_head = sh_head * (1 - .62 * smooth(.56, .76, y)) * (1 - .45 * smooth(.55, .95, np.abs(dx)))     # the hair takes none of it
    # never in focus, but not flat either: cloth falls away from the light and creases, and the neck is in the shadow of the chin
    fold = 1 + .17 * np.sin(x * 6.1 + y * 2.3 + 1.0) * np.sin(y * 4.7 - x * 1.3) + .10 * np.sin(x * 11.0 - y * 5.0)
    away = 1 - .62 * np.clip(np.abs(x - cx) / .95, 0, 1) ** 2
    chin = 1 - .62 * np.exp(-((x - cx) / .21) ** 2 - ((y - .03) / .17) ** 2)
    sh_body = (.16 + .58 * np.clip((y + 1.0) / .85, 0, 1) ** 1.3) * fold * away * chin
    sh_arm = (.52 + .30 * np.clip((y + 1.0) / 1.0, 0, 1)) * (1 - .3 * smooth(-.2, -1.0, y))
    shade = np.where(head < .004, sh_head, np.where(arms < .004, sh_arm, sh_body))
    return mask, shade, arms, solid


class Portrait:
    def __init__(self, gl: GL, size):
        self.gl = gl
        self.size = tuple(size)
        self.aspect = size[0] / size[1]
        self.t_cam = gl.target(size)
        self.cam_done = False
        self.t_form = None
        self.eye = EYE

    # ------------------------------------------------------------------ the camera in the glass
    def _render_camera(self, cam):
        a = camera_plate(self.gl, self.size, cam)
        self.t_cam.tex.write(a.astype("f2").tobytes())
        self.cam_alpha = np.flipud(a[..., 3])       # top row first
        self.cam_done = True

    # ------------------------------------------------------------------ the form
    def _set_form(self):
        W, H = self.size
        xs = (np.arange(W) + .5) / W * 2 * self.aspect - self.aspect
        ys = 1 - (np.arange(H) + .5) / H * 2
        x, y = np.meshgrid(xs, ys)
        mask, shade, arms, solid = silhouette(x, y)
        rgba = np.stack([solid * shade, solid, np.zeros_like(solid), np.ones_like(solid)], -1)
        rgba = (np.flipud(np.clip(rgba, 0, 1)) * 255).astype("u1")
        t = self.gl.ctx.texture((W, H), 4, rgba.tobytes(), dtype="f1")
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        self.t_form = t

    def fig(self, st, cam, out: Target):
        """one frame of what the glass holds, into out (rgb + how much it blocks what's behind)."""
        if not self.cam_done:
            self._render_camera(cam)
        if self.t_form is None:
            self._set_form()
        self.gl.run("desk_portrait.frag", out, uForm=self.t_form, uCam=self.t_cam, uAspect=self.aspect,
                    uBreath=st.get("breath", 0.0), uK=FORM, uCool=COOL)
        out.gen()
