"""The camera's information, handwritten. Same positions as film/hud.py, but the bars are the same paper as the
world and everything on them is ink: Xingkai with a brush's unevenness, brushed marks for the icons, one carved
vermilion seal. All wobble and dryness are seeded per mark, never per frame, so nothing crawls.

  draw(s) -> RGBA uint8 (768, 1024, 4), taking the same state dict as film/hud.py.
"""
from __future__ import annotations
import re
import zlib
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from . import fonts

S = 2           # supersample of the ink; the panel itself is 1024x768
W, H = 1024, 768
TOP, BOT = 96, 672
BASE = np.array((.905, .852, .742), "f4")       # the world's paper
K_INK = np.array((1.127, 1.15, 1.185), "f4")    # the world's extinctions
K_RED = np.array((.15, 1.22, 1.62), "f4")
INK_RGB = np.array((.035, .032, .033), "f4")    # ink and vermilion where they lie over the picture, not on paper
RED_RGB = np.array((.74, .17, .09), "f4")
INK, RED = 0, 1

DIGITS = "零一二三四五六七八九"
LARGER = {"八": 1.32, "六": 1.12, "二": 1.08, "三": 1.08}     # Xingkai writes these small; a brush at this size would not


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# ---- writing numbers ---------------------------------------------------------------------------
def zh_int(n: int, spoken=False) -> str:
    """6400 -> 六千四百, or 六千四 as it is said aloud."""
    n = int(n)
    if n == 0:
        return "零"
    out, zero = "", False
    for value, unit in ((10000, "万"), (1000, "千"), (100, "百"), (10, "十"), (1, "")):
        q = n // value
        n %= value
        if q == 0:
            zero = bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        out += (zh_int(q) if q >= 10 else DIGITS[q]) + unit
    if out.startswith("一十"):
        out = out[1:]
    if spoken and len(out) >= 3 and out[-1] in "百千十" and out[-3] in "万千百":
        out = out[:-1]
    return out


def zh_digits(s: str) -> str:
    return "".join(DIGITS[int(c)] if c.isdigit() else c for c in s)


def zh_float(s) -> str:
    """1.8 -> 一点八"""
    a, _, b = str(s).partition(".")
    return zh_int(int(a or 0)) + ("点" + zh_digits(b) if b else "")


def zh_shutter(s) -> str:
    s = str(s)
    m = re.fullmatch(r"1/(\d+)", s)
    if m:
        return zh_int(int(m.group(1))) + "分之一"
    m = re.fullmatch(r"([\d.]+)\"?", s)
    return zh_float(m.group(1)) + "秒" if m else s


def zh_fnum(s) -> str:
    m = re.search(r"[\d.]+", str(s))
    return zh_float(m.group(0)) if m else str(s)


def zh_iso(s) -> str:
    m = re.search(r"\d+", str(s))
    return zh_int(int(m.group(0)), spoken=True) if m else str(s)


def zh_ev(s) -> str:
    m = re.search(r"[\d.]+", str(s))
    if not m or float(m.group(0)) == 0:
        return "零"
    return ("减" if str(s)[0] in "-−" else "加") + zh_float(m.group(0))


def zh_remain(s) -> str:
    h = re.search(r"(\d+)\s*h", str(s))
    m = re.search(r"(\d+)\s*m", str(s))
    out = (zh_int(int(h.group(1))) + "时" if h else "") + (zh_int(int(m.group(1))) + "分" if m else "")
    return out or str(s)


def zh_tc(s) -> str:
    p = [int(x) for x in re.findall(r"\d+", str(s))][-3:]
    p = [0] * (3 - len(p)) + p
    return f"{zh_int(p[0])}时{zh_int(p[1])}分{zh_int(p[2])}秒"


def zh_timecode(s) -> str:
    """00:00:12.05 -> digit by digit, so that a running code keeps its width."""
    p = re.findall(r"\d+", str(s))
    return " ".join(zh_digits(x.zfill(2)) for x in p) or str(s)


def zh_format(s) -> str:
    return "超清" if "4K" in str(s) else "高清" if "HD" in str(s) else str(s)


def zh_fps(s) -> str:
    m = re.match(r"\s*(\d+)", str(s))
    if not m:
        return str(s)
    n = int(m.group(1))
    return ("廿" + (DIGITS[n - 20] if n > 20 else "") if 20 <= n < 30 else zh_int(n)) + "格"


def zh_mag(s) -> str:
    m = re.search(r"[\d.]+", str(s))
    return (zh_float(m.group(0)) if "." in m.group(0) else zh_int(int(m.group(0)))) + "倍" if m else str(s)


def zh_dist(s) -> str:
    m = re.fullmatch(r"\s*([\d.]+)\s*(mm|cm|m)?\s*", str(s))
    if not m:
        return str(s)
    return zh_float(m.group(1)) + {"mm": "毫", "cm": "厘", "m": "米", None: ""}[m.group(2)]


WORDS = {"AF-C": "对焦", "AF-S": "单对", "MF": "手对", "AWB": "白平衡", "ST": "标准", "PT": "肖像", "NT": "中性", "VV": "鲜明",
         "FL": "胶片", "IN": "清淡", "SH": "柔和", "BW": "黑白", "SE": "褐色", "M": "手", "A": "光", "S": "快", "P": "程"}


# ---- paper fibre for the brush to skip over --------------------------------------------------------
def _filtered(n, seed, ax, ay):
    rng = np.random.default_rng(seed)
    f = np.fft.rfft2(rng.standard_normal((n, n)))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    a = np.real(np.fft.irfft2(f * np.exp(-(fx * ax) ** 2 - (fy * ay) ** 2), s=(n, n)))
    # uniform, so that a threshold means a fraction of the paper
    r = np.argsort(np.argsort(a.ravel())).reshape(a.shape)
    return (r / (n * n)).astype("f4")


def _blur(a, r):
    """Gaussian of a float mask through PIL (8 bits are plenty for what it is used for)."""
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype("u1"))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), "f4") / 255


class HudInk:
    def __init__(self):
        n = 512
        self.n = n
        self.fib = _filtered(n, 11, 2.2, 7.0)       # short fibres
        self.fib = .5 * self.fib + .5 * _filtered(n, 12, 7.0, 2.0)
        self.streak = _filtered(n, 13, 26.0, 1.6)   # along the brush
        self.mottle = _filtered(n, 14, 60.0, 60.0)
        self.bold = "Xingkai.ttc"
        self.paper = self._numpy_paper()
        self._text_cache = {}
        self._stroke_cache = {}
        self._seal_cache = {}
        self._tick_cache = {}

    def _numpy_paper(self):
        """Used only until the world hands over its own paper."""
        rng = np.random.default_rng(3)
        m = _filtered(self.n, 15, 40.0, 40.0)[:H % self.n or None]
        a = np.tile(self.fib, (2, 3))[:H, :W] - .5
        b = np.tile(_filtered(self.n, 16, 110.0, 110.0), (2, 3))[:H, :W] - .5
        tone = a * .03 + b * .05
        _ = (m, rng)
        return (BASE * (1 + tone[..., None] * np.array((.9, 1., 1.25), "f4"))).astype("f4")

    def set_paper(self, rgb):
        """(768, 1024, 3) display-referred floats, from the world's own paper shader."""
        self.paper = np.asarray(rgb, "f4")

    def _noise(self, tile, x, y, w, h):
        n = self.n
        yy = (np.arange(h) + y) % n
        xx = (np.arange(w) + x) % n
        return tile[yy[:, None], xx[None, :]]

    # ---- writing ------------------------------------------------------------------------------
    def _text_mask(self, text, size, seed, load, dry, maxw):
        key = (text, size, seed, load, dry, maxw)
        hit = self._text_cache.get(key)
        if hit is not None:
            return hit
        rng = np.random.default_rng(seed)
        ss = 2                                          # glyphs are rotated at twice the ink's resolution
        px = int(round(size * S * ss))
        font = fonts.font(self.bold, px, 0)
        chars = list(text)
        tight = rng.uniform(.90, .97, len(chars))
        width = sum(font.getlength(c) * t for c, t in zip(chars, tight))
        if maxw and width / (S * ss) > maxw:
            k = maxw / (width / (S * ss))
            px = max(int(px * k), 8)
            font = fonts.font(self.bold, px, 0)
            width = sum(font.getlength(c) * t for c, t in zip(chars, tight))
        pad = px
        cw, ch = int(width + 2 * pad), int(px * 2.2)
        base = int(px * 1.45)
        img = Image.new("L", (cw, ch), 0)
        x = float(pad)
        ranges = []
        for c, t in zip(chars, tight):
            a = font.getlength(c)
            if c == " ":
                x += a * .45
                continue
            box = int(max(a * 1.7, px * 1.7)) + 8
            cell = Image.new("L", (box, box), 0)
            ImageDraw.Draw(cell).text((box / 2, box * .72), c, font=font, fill=255, anchor="ms")
            cell = cell.rotate(rng.uniform(-3.4, 3.4), resample=Image.BICUBIC)
            sc = rng.uniform(.93, 1.07) * LARGER.get(c, 1.0)
            nb = max(int(box * sc), 4)
            cell = cell.resize((nb, nb), Image.BICUBIC)
            ox = int(x + a / 2 - nb / 2)
            oy = int(base + rng.uniform(-.05, .05) * px - nb * .72)
            region = (ox, oy, ox + nb, oy + nb)
            img.paste(ImageChops.lighter(img.crop(region), cell), region)
            ranges.append((x / ss, (x + a) / ss, rng.uniform(.84, 1.10)))
            x += a * t
        # Xingkai's hairlines are thinner than a panel pixel: a loaded brush is not
        img = img.filter(ImageFilter.MaxFilter(3)).resize((cw // ss, ch // ss), Image.LANCZOS)
        a = np.asarray(img, "f4") / 255
        h_, w_ = a.shape
        ox, oy = int(seed % 397), int((seed >> 9) % 397)
        fib = self._noise(self.fib, ox, oy, w_, h_)
        streak = self._noise(self.streak, oy, ox, w_, h_)
        blur = _blur(a, .6)
        # the edge of a wet stroke is decided by the fibres under it
        cov = sstep(.24, .50, blur + (fib - .5) * .22)
        # one load of the brush per character, running out along the phrase
        lo = np.ones(w_, "f4") * .9
        for x0, x1, k in ranges:
            lo[int(x0):int(x1) + 1] = k
        run = 1 - .15 * np.linspace(0, 1, w_)
        lo = lo * run
        # dry brush where the load is low
        thin = np.clip(1.1 - lo, 0, 1)[None, :]
        cov *= 1 - sstep(.82 - .42 * thin * dry, .96 - .38 * thin * dry, streak) * dry * .8
        # ink pools at the edge of a stroke and thins at its heart
        heart = _blur(a, 1.5)
        d = cov * (lo[None, :] * 1.25 + .55 * np.clip(1 - heart, 0, 1)) * load
        # a faint bleed along the fibres
        d += _blur(a, 2.8) * (.25 + 1.1 * fib) * .10 * load
        out = (d.astype("f4"), pad // ss, base // ss, width / ss)
        self._text_cache[key] = out
        return out

    def _add(self, m, x, y, ch=INK, k=1.0):
        """Add a density mask whose top-left is at S-scaled x, y."""
        arr = self.d[ch]
        x, y = int(round(x)), int(round(y))
        h, w = m.shape
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W * S), min(y + h, H * S)
        if x1 <= x0 or y1 <= y0:
            return
        arr[y0:y1, x0:x1] += m[y0 - y:y1 - y, x0 - x:x1 - x] * k
        self.y0, self.y1 = min(self.y0, y0), max(self.y1, y1)

    def text(self, x, y, s, size, slot="", anchor="ls", ch=INK, load=1.0, dry=1.0, maxw=None, k=1.0, stable=False):
        """stable: a running value. The hand belongs to the slot, not to what it says, so unchanged glyphs do not shift."""
        if not s:
            return
        if len(self._text_cache) > 3000:
            self._text_cache.clear()
        m, pad, base, width = self._text_mask(s, size, seed_of(slot) if stable else seed_of(slot, s), load, dry, maxw)
        px = x * S - pad
        if anchor[0] == "m":
            px -= width / 2
        elif anchor[0] == "r":
            px -= width
        self._add(m, px, y * S - base, ch, k)

    # ---- a stroke of the brush: bristles, each with its own life ------------------------------------
    def _stroke_mask(self, pts, width, seed, dry, taper):
        ss = 2
        P = np.asarray(pts, "f4") * S * ss
        seg = np.hypot(*np.diff(P, axis=0).T)
        L = np.concatenate(([0], np.cumsum(seg)))
        total = max(L[-1], 1e-3)
        n = max(int(total / 5), 6)
        u = np.linspace(0, 1, n)
        x = np.interp(u * total, L, P[:, 0])
        y = np.interp(u * total, L, P[:, 1])
        tx, ty = np.gradient(x), np.gradient(y)
        tl = np.maximum(np.hypot(tx, ty), 1e-4)
        nx, ny = -ty / tl, tx / tl
        wpx = width * S * ss
        pad = int(wpx + 10)
        x0, y0 = int(np.floor(x.min())) - pad, int(np.floor(y.min())) - pad
        w, h = int(np.ceil(x.max())) + pad - x0, int(np.ceil(y.max())) + pad - y0
        rng = np.random.default_rng(seed)
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        pressure = (.62 + .38 * np.sin(np.pi * u ** .62)) if taper else np.ones(n)
        pressure *= 1 - taper * .55 * sstep(.72, 1., u)
        nb = max(int(wpx / 1.6), 2)
        lw = max(int(round(wpx / nb * 1.45)), 1)
        ph = rng.uniform(0, 6.28, nb)
        for i in range(nb):
            o = (i + .5) / nb - .5
            edge = abs(o) * 2
            # outer bristles lift first; a dry brush breaks up
            start = rng.uniform(0, .05) * (.3 + edge)
            end = 1 - rng.uniform(0, .24) * dry * (.25 + edge)
            on = (u >= start) & (u <= end)
            if dry > 0:
                gaps = np.sin(u * total / (S * ss) * rng.uniform(.10, .34) + ph[i]) + rng.uniform(-.2, .2) + 1.65 * (1 - dry * (.35 + .65 * edge * .8)) + .34
                on &= gaps > 0
            ox = x + nx * o * wpx * pressure - x0
            oy = y + ny * o * wpx * pressure - y0
            lev = int(255 * rng.uniform(.70, 1.))
            idx = np.flatnonzero(on)
            if len(idx) < 2:
                continue
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if len(run) >= 2:
                    d.line(list(zip(ox[run].tolist(), oy[run].tolist())), fill=lev, width=lw, joint="curve")
        img = img.resize((max(w // ss, 1), max(h // ss, 1)), Image.BOX).filter(ImageFilter.GaussianBlur(.42))
        return np.asarray(img, "f4") / 255, x0 / ss, y0 / ss

    def stroke(self, pts, width=2.0, slot="", ch=INK, k=1.0, dry=.6, taper=1.0, cache=True):
        key = (tuple(np.round(np.asarray(pts, "f4").ravel(), 2).tolist()), width, slot, dry, taper) if cache else None
        hit = self._stroke_cache.get(key) if cache else None
        if hit is None:
            hit = self._stroke_mask(pts, width, seed_of(slot), dry, taper)
            if cache:
                if len(self._stroke_cache) > 4000:
                    self._stroke_cache.clear()
                self._stroke_cache[key] = hit
        m, x0, y0 = hit
        self._add(m * 1.45, x0, y0, ch, k)

    def dab(self, x, y, rx, ry, slot="", ch=INK, k=1.0):
        """A touch of the brush's tip."""
        key = ("dab", round(rx * 4), round(ry * 4), slot)
        hit = self._tick_cache.get(key)
        if hit is None:
            rng = np.random.default_rng(seed_of(slot))
            w, h = int(rx * 2 * S + 8), int(ry * 2 * S + 8)
            yy, xx = np.mgrid[0:h, 0:w]
            a = np.arctan2((yy - h / 2) / ry, (xx - w / 2) / rx)
            rr = np.hypot((xx - w / 2 + .5) / (rx * S), (yy - h / 2 + .5) / (ry * S))
            edge = 1 + sum(rng.uniform(.03, .10) * np.sin(a * f + rng.uniform(0, 6.28)) for f in (2, 3, 5, 9))
            fib = self._noise(self.fib, rng.integers(400), rng.integers(400), w, h)
            m = sstep(1.05, .78, rr / edge + (fib - .5) * .32).astype("f4")
            hit = m * (.95 + .5 * sstep(.45, 1., rr))
            self._tick_cache[key] = hit
        h, w = hit.shape
        self._add(hit, x * S - w / 2, y * S - h / 2, ch, k)

    # ---- the seal: carved, not typed ---------------------------------------------------------------
    def seal(self, cx, cy, ch="候", size=36, rot=-2.2):
        key = (ch, size, rot)
        hit = self._seal_cache.get(key)
        if hit is None:
            hit = self._carve(ch, size, rot)
            self._seal_cache[key] = hit
        h, w = hit.shape
        self._add(hit, cx * S - w / 2, cy * S - h / 2, RED)

    def _carve(self, ch, size, rot):
        ss = 4
        n = int(size * S * ss)
        pad = n // 5
        N = n + 2 * pad
        rng = np.random.default_rng(seed_of("seal", ch))
        u = np.linspace(0, 1, 9)
        # the stone's outline is not quite square
        corners = []
        for x0, y0, x1, y1 in ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)):
            for t in u[:-1]:
                j = rng.normal(0, .010, 2)
                corners.append((pad + (x0 + (x1 - x0) * t + j[0]) * n, pad + (y0 + (y1 - y0) * t + j[1]) * n))
        img = Image.new("L", (N, N), 0)
        ImageDraw.Draw(img).polygon(corners, fill=255)
        body = np.asarray(img, "f4") / 255
        # the character is what the knife took away
        font = fonts.font("Songti.ttc", int(n * .80), 6)
        gi = Image.new("L", (N, N), 0)
        ImageDraw.Draw(gi).text((N / 2, N / 2 + n * .01), ch, font=font, fill=255, anchor="mm")
        gi = gi.filter(ImageFilter.MaxFilter(2 * (n // 90) + 1))
        # knife strokes wander: push the glyph through a little displacement
        g = np.asarray(gi, "f4") / 255
        nx = self._noise(self.mottle, 31, 7, N, N) - .5
        ny = self._noise(self.mottle, 91, 201, N, N) - .5
        yy, xx = np.mgrid[0:N, 0:N]
        amp = n * .030
        g = g[np.clip((yy + ny * amp).astype(int), 0, N - 1), np.clip((xx + nx * amp).astype(int), 0, N - 1)]
        g = _blur(g, n / 110)
        fine = self._noise(self.fib, 5, 77, N, N)
        carved = sstep(.42, .60, g + (fine - .5) * .22)
        ink = body * (1 - carved)
        # broken edges: the rim has crumbled, more in some places than others
        dist = np.minimum.reduce([xx - pad, N - pad - xx, yy - pad, N - pad - yy]).astype("f4") / n
        big = self._noise(self.mottle, 300, 40, N, N)
        rim = .018 + .075 * sstep(.62, .92, big)
        ink *= sstep(0, .012, dist - rim * (.4 + 1.1 * fine))
        # a few real chips where the stone gave way
        chips = Image.new("L", (N, N), 0)
        cd = ImageDraw.Draw(chips)
        for _ in range(9):
            side = rng.integers(4)
            t = rng.uniform(.05, .95)
            cx, cy = [(t, 0), (1, t), (t, 1), (0, t)][side]
            cx, cy = pad + cx * n, pad + cy * n
            r = rng.uniform(.035, .10) * n
            pts = [(cx + np.cos(a) * r * rng.uniform(.5, 1.2), cy + np.sin(a) * r * rng.uniform(.5, 1.2))
                   for a in np.linspace(0, 6.28, 7, endpoint=False)]
            cd.polygon(pts, fill=255)
        ink *= 1 - np.asarray(chips, "f4") / 255
        # cinnabar paste takes unevenly
        paste = self._noise(self.mottle, 12, 140, N, N)
        ink *= 1 - sstep(.80, .93, paste * .55 + fine * .45) * .85
        d = ink * (1.05 + .55 * paste)
        im = Image.fromarray((np.clip(d / 2, 0, 1) * 255).astype("u1")).rotate(rot, resample=Image.BICUBIC)
        im = im.resize((N // ss, N // ss), Image.LANCZOS)
        return np.asarray(im, "f4") / 255 * 2

    # ---- the screen -------------------------------------------------------------------------------
    def draw(self, s: dict) -> np.ndarray:
        self.d = [np.zeros((H * S, W * S), "f4") for _ in range(2)]
        self.pa = None                                  # paper patches over the picture
        self.y0, self.y1 = H * S, 0
        vis = s.get("visible", True)
        if vis:
            self._bars(s)
            if s.get("level") is not None:
                self._level(s["level"])
            if s.get("hist") is not None and s["hist"] is not True:
                self._hist(s["hist"], s.get("hist_k", 1.0))
            if s.get("mag"):
                self._mag(s["mag"])
            if s.get("menu"):
                self._menu(s["menu"])
        if s.get("af"):
            self._af(s["af"])
        if s.get("banner"):
            self._banner(s["banner"])
        return self._compose(vis)

    def _compose(self, bars):
        out = np.zeros((H, W, 4), "u1")
        dI, dR = self.d
        # transmission is averaged down, so that ink keeps its own edge
        t = np.exp(-dI[..., None] * K_INK - dR[..., None] * K_RED)
        t = (t[::2, ::2] + t[1::2, ::2] + t[::2, 1::2] + t[1::2, 1::2]) * .25
        a_ink = 1 - np.exp(-(dI * 1.15 + dR * 1.0))
        a_ink = (a_ink[::2, ::2] + a_ink[1::2, ::2] + a_ink[::2, 1::2] + a_ink[1::2, 1::2]) * .25
        share = dR / np.maximum(dI + dR, 1e-4)
        share = (share[::2, ::2] + share[1::2, ::2] + share[::2, 1::2] + share[1::2, 1::2]) * .25
        col = INK_RGB * (1 - share[..., None]) + RED_RGB * share[..., None]
        pa = np.zeros((H, W), "f4")
        if bars:
            pa[:TOP] = 1
            pa[BOT:] = 1
        if self.pa is not None:
            pa = np.maximum(pa, self.pa)
        pm = (self.paper * t) * pa[..., None] + col * (a_ink * (1 - pa))[..., None]
        alpha = pa + (1 - pa) * a_ink
        rgb = pm / np.maximum(alpha[..., None], 1e-3)
        out[..., :3] = (np.clip(rgb, 0, 1) * 255 + .5).astype("u1")
        out[..., 3] = (np.clip(alpha, 0, 1) * 255 + .5).astype("u1")
        return out

    def _patch(self, x0, y0, x1, y1, a=1.0, feather=5.0, slot="patch"):
        """A piece of the same paper laid over the picture, torn soft at the edges."""
        if self.pa is None:
            self.pa = np.zeros((H, W), "f4")
        xi0, yi0, xi1, yi1 = int(x0 - feather * 2), int(y0 - feather * 2), int(x1 + feather * 2), int(y1 + feather * 2)
        xi0, yi0, xi1, yi1 = max(xi0, 0), max(yi0, 0), min(xi1, W), min(yi1, H)
        yy, xx = np.mgrid[yi0:yi1, xi0:xi1]
        n = self._noise(self.fib, seed_of(slot) % 400, 33, xi1 - xi0, yi1 - yi0) - .5
        dist = np.minimum.reduce([xx - x0, x1 - xx, yy - y0, y1 - yy]).astype("f4") + n * feather * 1.6
        self.pa[yi0:yi1, xi0:xi1] = np.maximum(self.pa[yi0:yi1, xi0:xi1], sstep(-feather, feather * .5, dist) * a)

    def _box(self, x0, y0, x1, y1, width, slot, ch=INK, k=1.0, dry=.5, over=1.5):
        """Four strokes in the order a hand makes them, each overshooting its corner a little."""
        self.stroke([(x0 - over, y0), (x1 + over, y0 + .4)], width, slot + "t", ch, k, dry)
        self.stroke([(x1, y0 - over), (x1 - .3, y1 + over)], width, slot + "r", ch, k, dry)
        self.stroke([(x0 - over, y1), (x1 + over, y1 - .3)], width, slot + "b", ch, k, dry)
        self.stroke([(x0, y0 - over), (x0 + .3, y1 + over)], width, slot + "l", ch, k, dry)

    def _bars(self, s):
        t = s.get("t", 0.0)
        # ---- top row: audio, focus mode, standby and time, format, card, battery
        y = 54
        # audio: dabs, as many as the sound is loud
        al, ar = s.get("audio", (0.1, 0.1))
        for i, (lab, lv) in enumerate((("左", al), ("右", ar))):
            yy = 35 + i * 13
            self.text(26, yy + 8, lab, 12, "aud" + lab, load=.7)
            for k_ in range(26):
                on = k_ / 26 < lv
                hot = on and k_ > 22
                self.dab(42 + k_ * 7 + 2, yy + 2.5, 2.3, 3.6, f"a{i}{k_}", RED if hot else INK, 1.05 if on else .10)
        self.text(246, y, WORDS.get(s.get("afm", "AF-C"), s.get("afm", "AF-C")), 23, "afm", maxw=70)
        if s.get("rec"):
            self.seal(440, 45, "录", 40, 1.4)
        else:
            self.seal(440, 45, "候", 37)
        self.text(490, y, zh_tc(s.get("tc", "0:00:00")), 20, "tc", maxw=124, stable=True)
        self.text(620, y, zh_format(s.get("codec", "4K")), 18, "format", maxw=36)
        self.text(660, y, zh_fps(s.get("fmt", "24p")), 18, "fps", maxw=56)
        card = [(722, 32), (738, 32), (743, 37), (743, 58), (722, 58), (722, 31)]
        self.stroke(card, 1.8, "card", k=.9, dry=.35, taper=.4)
        self.text(732.5, 52, "一", 16, "card", anchor="ms")
        self.text(752, y, zh_remain(s.get("remain", "2h 14m")), 20, "remain", maxw=96)
        # battery: a brushed glyph, filled as far as the ink has gone
        bx = 926
        batt = float(s.get("battery", 84))
        self._box(bx, 35, bx + 48, 57, 2.0, "batt", k=.95, dry=.4)
        self.dab(bx + 51, 46, 2.2, 5.0, "battnub")
        run = 3 + 42 * batt / 100
        self.stroke([(bx + 3, 46.5), (bx + run, 45.8)], 13.0, "battfill", k=.86, dry=.75, taper=.25)
        self.text(bx - 8, y, zh_int(int(batt)), 21, "battpct", anchor="rs", maxw=66)
        # ---- second row: settings shown while on standby
        y = 85
        self._box(26, 66, 62, 91, 2.0, "mode", k=.92)
        self.text(44, 87, WORDS.get(s.get("mode", "M"), s.get("mode", "M")), 24, "mode", anchor="ms")
        for x0, y0, dx, dy in ((86, 67, 1, 1), (112, 67, -1, 1), (86, 89, 1, -1), (112, 89, -1, -1)):
            self.stroke([(x0, y0 + 7 * dy), (x0, y0), (x0 + 7 * dx, y0)], 1.8, f"afi{x0}{y0}", dry=.3, taper=.7)
        self.dab(99, 78, 2.6, 3.2, "afidot")
        look = str(s.get("look", "ST"))
        self.text(136, y, WORDS.get(look, look), 22 if len(WORDS.get(look, look)) == 1 else 18, "look", maxw=46)
        self.text(202, y + 1, "稳", 22, "steady", anchor="ms")
        # ---- bottom bar
        y = 706
        fi = s.get("focus", "off")
        cx, cy = 42, 697
        if fi == "lock":
            self.dab(cx, cy, 9.5, 9.0, "focus", k=1.4)
        elif fi == "hunt":
            if int(t * 6) % 2 == 0:
                self.dab(cx, cy, 9.5, 9.0, "focus", k=1.1)
        a = np.linspace(-.5, 5.55, 26)
        ring = [(cx + np.cos(q) * 10.5, cy + np.sin(q) * 10.2) for q in a]
        self.stroke(ring, 2.0, "focusring", k=.58 if fi == "off" else .8, dry=.65)
        self.text(86, y, zh_shutter(s.get("shutter", "1/50")), 25, "shutter", maxw=114)
        self.text(206, y, zh_fnum(s.get("fnum", "F1.8")), 28, "fnum", maxw=92)
        # exposure scale
        x0, x1, ym = 322, 622, 696
        for i in range(13):
            xx = x0 + (x1 - x0) * i / 12
            hh = 10 if i % 2 == 0 else 5.5
            self.stroke([(xx, ym), (xx + .4, ym - hh)], 1.9 if i % 6 == 0 else 1.5, f"ev{i}", k=.86, dry=.25)
        self.text(x0 - 14, ym + 1, "减", 15, "evm", anchor="ms", load=.8)
        self.text(x1 + 14, ym + 1, "加", 15, "evp", anchor="ms", load=.8)
        ev = float(np.clip(s.get("ev_meter", 0.0), -3, 3))
        xp = (x0 + x1) / 2 + ev / 3 * (x1 - x0) / 2
        self.stroke([(xp + .3, ym + 13), (xp, ym + 3)], 8.0, "evmark", RED, 1.05, dry=.15, taper=1.0, cache=False)
        self.text(660, y, zh_ev(s.get("ev", "±0.0")), 28, "ev", maxw=106)
        self.text(772, y, "感光", 17, "iso", maxw=36)
        self.text(808, y, zh_iso(s.get("iso", "6400")), 28, "isov", maxw=112)
        self.text(998, y, WORDS.get(s.get("wb", "AWB"), s.get("wb", "AWB")), 21, "wb", anchor="rs", maxw=72)
        # ---- lower line, in a paler ink
        self.text(86, 735, zh_timecode(s.get("timecode", "00:00:00.00")), 16, "timecode", load=.62, maxw=156, stable=True)
        self.text(250, 735, zh_lens(s.get("lens", "55mm")), 17, "lens", load=.62, maxw=70)

    def _af(self, a):
        px = a["x"] * W
        py = TOP + a["y"] * (BOT - TOP)
        w, h = a["w"], a["h"]
        al = float(a.get("alpha", 1.0))
        if al <= .01:
            return
        state = a.get("state", "search")
        ch = RED if state == "fail" else INK
        k = al * {"dim": .55, "search": 1.0, "track": 1.15, "lock": 1.35, "fail": 1.3}.get(state, 1.0)
        lw = float(a.get("lw", 2.0)) * 1.15
        x0, x1, y0, y1 = px - w / 2, px + w / 2, py - h / 2, py + h / 2
        self._wobbly(x0, y0, x1, y1, lw, ch, k, bool(a.get("corners")))
        if a.get("label"):
            self.text(x0, y0 - 7, str(a["label"]), 17, "aflabel", ch=ch, k=al)

    def _wobbly(self, x0, y0, x1, y1, lw, ch, k, corners):
        """The wobble is a function of position along each side, and belongs to the box, not to the frame."""
        rng = np.random.default_rng(1729)
        sides = ((x0 - 2.2, y0, x1 + 2.6, y0 + .6), (x1, y0 - 2.4, x1 - .5, y1 + 2.8), (x1 + 2.2, y1, x0 - 2.6, y1 - .5), (x0, y1 + 2.4, x0 + .5, y0 - 2.8))
        for i, (ax, ay, bx, by) in enumerate(sides):
            f1, f2 = rng.uniform(.8, 1.6), rng.uniform(2.1, 3.7)
            p1, p2 = rng.uniform(0, 6.28, 2)
            amp = rng.uniform(1.0, 1.7)
            u = np.linspace(0, 1, 15)
            e = amp * (np.sin(u * 6.283 * f1 + p1) * .72 + np.sin(u * 6.283 * f2 + p2) * .28) * np.sin(np.pi * np.minimum(u * 1.1, 1)) ** .5
            dx, dy = bx - ax, by - ay
            ln = max(np.hypot(dx, dy), 1e-3)
            nx, ny = -dy / ln, dx / ln
            if corners:
                c = .28
                for lo, hi, tag in ((0, c, "a"), (1 - c, 1, "b")):
                    v = np.linspace(lo, hi, 8)
                    ee = np.interp(v, u, e)
                    self.stroke(list(zip(ax + dx * v + nx * ee, ay + dy * v + ny * ee)), lw, f"af{i}{tag}", ch, k, dry=.42, cache=False)
            else:
                pts = list(zip(ax + dx * u + nx * e, ay + dy * u + ny * e))
                self.stroke(pts, lw, f"af{i}", ch, k, dry=.5, cache=False)

    def _hist(self, h, k=1.0):
        """A brushed silhouette: a ridge line loaded with ink, and a wash that the mist takes at its foot."""
        x1, y1 = 998, 660
        w, ht = 150, 72
        x0, y0 = x1 - w, y1 - ht
        if k <= .01:
            return
        self._patch(x0 - 2, y0 + 6, x1 + 2, y1 + 2, a=.62 * k, feather=7, slot="hist")
        h = np.asarray(h, "f4")
        h = np.convolve(h, (.25, .5, .25), "same")
        h = h / max(h.max(), 1e-5)
        wp, hp = w * S, ht * S
        xs = np.linspace(0, len(h) - 1, wp)
        top = np.interp(xs, np.arange(len(h)), h)
        # a wandering hand
        n = self._noise(self.mottle, 50, 80, wp, 1)[0] - .5
        n2 = self._noise(self.fib, 10, 50, wp, 1)[0] - .5
        ytop = hp - (top * (ht - 3) * S) + n * 3.0 + n2 * 1.5
        yy = np.arange(hp)[:, None].astype("f4")
        depth = yy - ytop[None, :]
        inside = sstep(-.8, .8, depth)
        fib = self._noise(self.fib, 200, 11, wp, hp)
        streak = self._noise(self.streak, 7, 91, hp, wp).T
        fall = np.exp(-np.maximum(depth, 0) / (ht * S * .28))
        line = np.exp(-np.maximum(depth, 0) / 3.5)
        d = inside * (fall * (.55 + .55 * sstep(.25, .85, streak)) + line * .9) * (.86 + .28 * fib)
        d *= sstep(0, hp * .16, hp - yy)                      # the mist at its foot
        self._add(d.astype("f4"), x0 * S, y0 * S, INK, 1.25 * k)

    def _level(self, roll):
        cx, cy = W / 2, (TOP + BOT) / 2
        c, s = np.cos(roll), np.sin(roll)
        for r0, r1, tag in ((60, 300, "r"), (-60, -300, "l")):
            self.stroke([(cx + r0 * c, cy + r0 * s), (cx + r1 * c, cy + r1 * s)], 2.1, "lev" + tag, k=.9, dry=.55, cache=False)

    def _mag(self, m):
        nw, nh = 118, 66
        x1, y1 = 998, 656
        x0, y0 = x1 - nw, y1 - nh
        self._patch(x0, y0, x1, y1, a=.7, feather=4, slot="mag")
        self._box(x0, y0, x1, y1, 1.9, "mag", dry=.5)
        rx, ry, rz = m["x"], m["y"], m["zoom"]
        rw, rh = max(nw / rz, 3), max(nh / rz, 3)
        cx, cy = x0 + rx * nw, y0 + ry * nh
        self.dab(cx, cy, rw / 2 + 1, rh / 2 + 1, "magdot", RED, 1.1)
        self.text(x0 - 12, y1 - 6, zh_mag(m.get("label", "")), 26, "maglabel", anchor="rs", maxw=120, stable=True)

    def _menu(self, m):
        x0, y0 = 40, TOP + 26
        items = m["items"]
        row = 42
        x1, y1 = x0 + 176, y0 + 18 + row * len(items)
        self._patch(x0, y0, x1, y1, a=1.0, feather=5, slot="menu")
        self._box(x0, y0, x1, y1, 1.9, "menu", dry=.55)
        self.text(x0 + 12, y0 - 6, str(m.get("title", "")), 15, "menut", maxw=150)
        for i, it in enumerate(items):
            yy = y0 + 12 + i * row
            if i == m["sel"]:
                self.stroke([(x0 + 6, yy + row / 2 - 1), (x1 - 6, yy + row / 2 - 1.5)], row - 6.0, "menusel", RED, .55, dry=.7, taper=.2)
            code = str(it["code"])
            self.text(x0 + 18, yy + 29, WORDS.get(code, code), 24, f"menuc{i}", maxw=44)
            if it.get("name"):
                self.text(x0 + 62, yy + 28, str(it["name"]), 15, f"menun{i}", load=.75, maxw=106)

    def _banner(self, b):
        cx, cy = W / 2, (TOP + BOT) / 2
        self._patch(cx - 280, cy - 44, cx + 280, cy + 44, a=1.0, feather=6, slot="banner")
        self._box(cx - 276, cy - 40, cx + 276, cy + 40, 2.1, "banner", dry=.55)
        self.text(cx, cy + 10, str(b), 28, "banner", anchor="ms", maxw=520)


def zh_lens(s) -> str:
    m = re.search(r"\d+", str(s))
    return zh_int(int(m.group(0))) + "毫" if m else str(s)


_hud = None


def get() -> HudInk:
    global _hud
    if _hud is None:
        _hud = HudInk()
    return _hud


def draw(s: dict) -> np.ndarray:
    return get().draw(s)
