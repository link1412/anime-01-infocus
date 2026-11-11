"""The room. A monitor, a dark desk, a machine with a small white light.

Everything visible is computed from rays: the screen is a plane with a real pixel
grid, so filming it beats against the camera's own sampling.
"""
from __future__ import annotations
import numpy as np
import moderngl
from PIL import Image, ImageDraw
from ..core import GL, Target
from .. import fonts
from .flower import K_COC
from .desk_bake import Baked
from .desk_window import window_target

SCREEN_W, SCREEN_H = .597, .336       # metres: a 27-inch panel
RES = (2560, 1440)
TAN = .205                            # vertical half tangent of the 55 mm lens in 16:9
BG = (13, 13, 15)
LED = (.300, -.2005, -.0525)          # the machine's light, in the room
LED_R, LED_I = .0011, 1.0
LEAK, LEAK_FEED, DRIVE = 5.0, 2.6, 6.0   # the backlight, as the room sees it: wide open in the dark, less once the camera stops down. and how much the image adds
BAKE_BATCHES = 192


def is_cjk(ch):
    return ord(ch) > 0x2E7F


class Term:
    """Character cell terminal. CJK takes two cells."""
    FS = 56
    ADV = 33.7
    LINE = 86
    X0, Y0 = 110, 96

    def __init__(self):
        self.mono = fonts.font("Menlo.ttc", self.FS, 0)
        self.bold = fonts.font("Menlo.ttc", self.FS, 1)
        self.zh = fonts.font("Songti.ttc", self.FS, 6)
        self._last = None
        self._img = None

    def cell(self, row, col):
        return self.X0 + col * self.ADV, self.Y0 + row * self.LINE

    def char_positions(self, row, text, col0=0):
        """screen-pixel centres of each char of `text` on `row`."""
        out, col = [], col0
        for ch in text:
            w = 2 if is_cjk(ch) else 1
            x, y = self.cell(row, col)
            out.append((ch, x + w * self.ADV / 2, y + self.FS * .55))
            col += w
        return out

    def raster(self, lines, cursor=None, cursor_rgb=(205, 205, 205)):
        """lines: list of (text, rgb) or list of spans [(text, rgb), ...]. cursor: (row, col)."""
        key = repr((lines, cursor, cursor_rgb))
        if key == self._last:
            return self._img, False
        im = Image.new("RGB", RES, BG)
        d = ImageDraw.Draw(im)
        for row, ln in enumerate(lines):
            spans = ln if isinstance(ln, list) else [ln]
            col = 0
            for text, rgb in spans:
                for ch in text:
                    w = 2 if is_cjk(ch) else 1
                    x, y = self.cell(row, col)
                    if ch != " ":
                        if is_cjk(ch):
                            d.text((x + self.ADV, y + self.FS * .55), ch, font=self.zh, fill=rgb, anchor="mm")
                        else:
                            d.text((x, y), ch, font=self.mono, fill=rgb)
                    col += w
        if cursor:
            x, y = self.cell(*cursor)
            d.rectangle([x, y - 2, x + self.ADV, y + self.FS + 8], fill=cursor_rgb)
        self._last = key
        self._img = np.flipud(np.asarray(im, "u1"))
        return self._img, True


def camera(st):
    """camera position and basis. x right, y up, z out of the screen toward the operator."""
    d = st.get("dist", 1.1)
    o = np.array([st.get("ox", .012), st.get("oy", -.006), d], "f4")
    yaw, pitch, roll = st.get("yaw", 0.0), st.get("pitch", 0.0), st.get("roll", 0.0)
    f = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)], "f4")
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c, s = np.cos(roll), np.sin(roll)
    return o, f, r * c + u * s, u * c - r * s


def project(sx, sy, st, aspect):
    """screen pixel -> view units under camera st (used to land type exactly on the terminal)."""
    o, f, r, u = camera(st)
    p = np.array([(sx / RES[0] - .5) * SCREEN_W, (.5 - sy / RES[1]) * SCREEN_H, 0.0])
    d = p - o
    z = d @ f
    return (d @ r) / z / TAN, (d @ u) / z / TAN


class _Pass:
    """several attachments, one draw."""

    def __init__(self, fbo):
        self.fbo = fbo


class Desk:
    def __init__(self, gl: GL, size):
        self.gl = gl
        self.size = tuple(size)
        self.aspect = size[0] / size[1]
        self.term = Term()
        self.t_screen = gl.ctx.texture(RES, 3, dtype="f1")
        self.t_screen.build_mipmaps()
        self.t_screen.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_screen.repeat_x = self.t_screen.repeat_y = False
        self.t_scene = gl.target(size, mips=True)
        self.t_room = gl.target(size, nearest=True)
        self.mrt = _Pass(gl.ctx.framebuffer([self.t_scene.tex, self.t_room.tex]))
        self.t_drive = gl.target((2, 2), nearest=True)
        tile = (size[0] // 16 + 1, size[1] // 16 + 1)
        self.t_tile0 = gl.target(tile, nearest=True)
        self.t_tile = gl.target(tile, nearest=True)
        self.t_fig = gl.target(size, mips=True)
        self.portrait = None
        self.t_blank = gl.target((4, 4))
        self.window = None
        self.baked = None
        self.bake_batches = BAKE_BATCHES

    def set_lines(self, lines, cursor=None, cursor_rgb=(205, 205, 205)):
        img, changed = self.term.raster(lines, cursor, cursor_rgb)
        if changed:
            self.t_screen.write(img.tobytes(), alignment=1)
            self.t_screen.build_mipmaps()

    def make_window(self):
        """the night behind whoever is holding the camera, computed once when something first looks at it."""
        self.window = window_target(self.gl, self.size)

    def _led(self, st, o, f, r, u, zoom, fd):
        """where the machine's light lands, how large its disc is, whether the monitor is in the way."""
        lp = np.array(LED, "f4")
        v = lp - o
        z = float(v @ f)
        tan = TAN / zoom
        ppu = self.size[1] / 2
        vx, vy = float(v @ r) / z / tan, float(v @ u) / z / tan
        c = (.5 + vx / (2 * self.aspect), .5 + vy / 2)
        coc = K_COC * abs(fd - 1 / z) * zoom * ppu
        r0 = LED_R / (z * tan) * ppu
        rad = max(coc, r0)
        h = o + v * (float(o[2]) / (float(o[2]) - LED[2]))
        qx, qy = abs(h[0]) - .307, abs(h[1] + .00325) - .17975
        edge = float(np.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0))
        vis = float(np.clip(.5 + edge * ppu / (float(o[2]) * tan) / max(2 * rad, 1), 0, 1))
        k = st.get("led", 1.0) * vis * (r0 / rad) ** 2 * LED_I
        return (c[0], c[1], rad), (k * 1.0, k * .97, k * .92)

    def render(self, st: dict, out: Target, feed=None, bg=None):
        gl = self.gl
        if self.baked is None:
            self.baked = Baked(gl, batches=self.bake_batches)
        if st.get("lines") is not None and feed is None:
            self.set_lines(st["lines"], st.get("cursor"), st.get("cursor_rgb", (205, 205, 205)))
        o, f, r, u = camera(st)
        zoom = st.get("zoom", 1.0)
        cam = dict(uO=tuple(o), uF=tuple(f), uR=tuple(r), uU=tuple(u), uTan=TAN / zoom, uAspect=self.aspect,
                   uW=SCREEN_W, uH=SCREEN_H)
        scr = feed if feed is not None else self.t_screen
        fd = st["fd"]
        d = float(o[2])
        px_per_unit = self.size[1] / 2
        coc_s = K_COC * abs(fd - 1 / d) * zoom * px_per_unit
        coc_r = K_COC * abs(fd - 1 / (2 * d)) * zoom * px_per_unit
        # the beat between grid and sensor only exists while the grid is resolved
        beat = 1 - np.clip((coc_s - 1.0) / 5.0, 0, 1)
        is_feed = 1.0 if feed is not None else 0.0
        bright = st.get("bright", 1.0)
        # the frame's light
        gl.run("desk_drive.frag", self.t_drive, uScreen=scr, uIsFeed=is_feed, uBright=bright, uLeak=LEAK_FEED if feed is not None else LEAK, uGain=DRIVE)
        gl.run("desk_scene.frag", self.mrt, uScreen=scr, uIsFeed=is_feed, uBright=bright,
               uMoire=st.get("moire", 1.0) * beat, uRes=self.size, uT=st["t"], uGrid=st.get("grid", 1.0),
               uGlow=st.get("glow", 1.0), uLed=st.get("led", 1.0), uDrive=self.t_drive, uMask=8191,
               **self.baked.uniforms, **cam)
        self.t_scene.gen()
        kc = K_COC * zoom * px_per_unit
        id_g = 1 / d
        gl.run("desk_tile.frag", self.t_tile0, uRoom=self.t_room, uPrev=self.t_blank, uRoomSize=self.size, uKc=kc, uFd=fd,
               uIdG=id_g, uDilate=0.0)
        gl.run("desk_tile.frag", self.t_tile, uRoom=self.t_room, uPrev=self.t_tile0, uRoomSize=self.size, uKc=kc, uFd=fd,
               uIdG=id_g, uDilate=1.0)
        refl = st.get("refl", 0.0)
        if refl > 0:
            self.portrait.fig(st, dict(uO=cam["uO"], uF=cam["uF"], uR=cam["uR"], uU=cam["uU"], uTan=cam["uTan"], uAspect=cam["uAspect"]), self.t_fig)
        led_uv, led_rgb = self._led(st, o, f, r, u, zoom, fd)
        gl.run("desk_comp.frag", out, uScene=self.t_scene, uRoom=self.t_room, uTile=self.t_tile,
               uFig=self.t_fig if refl > 0 else self.t_blank, uRefl=refl,
               uCocS=coc_s, uCocR=coc_r, uPx=(1 / self.size[0], 1 / self.size[1]), uKc=kc, uFd=fd, uIdG=id_g,
               uLedUV=led_uv, uLedRGB=led_rgb, uO=cam["uO"], uF=cam["uF"], uR=cam["uR"], uU=cam["uU"], uTan=cam["uTan"], uAspect=cam["uAspect"],
               uBg=bg if bg is not None else self.t_blank, uHasBg=1.0 if bg is not None else 0.0,
               uWindow=self.window if self.window is not None else self.t_blank, uWin=st.get("win", 0.0))

    def lookdev(self, t):
        lines = [("> 雨中的花。花瓣上的水珠。水珠里的海。", (205, 205, 205)), ("", (0, 0, 0)),
                 ("$ grep -rn hello film/", (140, 200, 150))]
        fd = 1 / 1.1
        if t > 72:
            fd = 1 / 2.2
        st = dict(t=t, fd=fd, dist=1.1, lines=lines, cursor=(3, 0), refl=.14 if t > 70 else 0.0, look=1.0)
        return {"world": st, "isp": dict(ev=.8, iso=.6), "hud": dict(shutter="1/50", fnum="F1.8", iso="3200"), "ph": dict(scale=1.1)}
