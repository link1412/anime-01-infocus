"""EVF information display. Layout language follows the A7-series movie screen
(16:9 image with info bars above and below). No logos or brand names."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from . import fonts

S = 2          # supersample, the panel itself is 1024x768
W, H = 1024, 768
TOP, BOT = 96, 672


def _font(name, size, index=0):
    return fonts.font(name, size * S, index)


class Hud:
    def __init__(self):
        self.big = _font("HelveticaNeue.ttc", 31, 4)       # condensed bold
        self.med = _font("HelveticaNeue.ttc", 23, 4)
        self.small = _font("Arial Narrow.ttf", 21)
        self.tiny = _font("Arial Narrow.ttf", 17)
        self.zh = _font("PingFang.ttc", 24, 7)

    # ---- primitives (panel px) ------------------------------------------------
    def _t(self, d, x, y, s, f, c=(255, 255, 255, 255), anchor="ls", shadow=True):
        if shadow:
            d.text(((x + 1) * S, (y + 1) * S), s, font=f, fill=(0, 0, 0, min(c[3], 170)), anchor=anchor)
        d.text((x * S, y * S), s, font=f, fill=c, anchor=anchor)

    def _r(self, d, x0, y0, x1, y1, outline=None, fill=None, w=1.5):
        d.rectangle([x0 * S, y0 * S, x1 * S, y1 * S], outline=outline, fill=fill, width=int(w * S))

    def _l(self, d, pts, c, w=1.5):
        d.line([(x * S, y * S) for x, y in pts], fill=c, width=int(w * S))

    # ---- full screen ----------------------------------------------------------
    def draw(self, s: dict) -> np.ndarray:
        img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        if s.get("visible", True):
            self._bars(d, s)
            if s.get("level") is not None:
                self._level(d, s["level"])
            if s.get("hist") is not None:
                self._hist(d, s["hist"], s)
            if s.get("mag"):
                self._mag(d, s["mag"])
            if s.get("menu"):
                self._menu(d, s["menu"])
        if s.get("af"):
            self._af(d, s["af"])
        if s.get("banner"):
            self._banner(d, s["banner"])
        out = img.resize((W, H), Image.LANCZOS)
        return np.asarray(out, dtype="u1")

    def _bars(self, d, s):
        white = (255, 255, 255, 255)
        dim = (196, 196, 196, 255)
        t = s.get("t", 0.0)
        # ---- area 1 (top): audio, focus mode, STBY/REC and time, format, card, battery
        y = 54
        # audio meters: real levels from the phone-room mix
        al, ar = s.get("audio", (0.1, 0.1))
        for i, (lab, lv) in enumerate((("L", al), ("R", ar))):
            yy = 35 + i * 13
            self._t(d, 26, yy + 7, lab, self.tiny, dim, shadow=False)
            for k in range(26):
                on = k / 26 < lv
                c = (255, 255, 255, 235) if on else (86, 86, 86, 255)
                if on and k > 22:
                    c = (255, 80, 60, 255)
                self._r(d, 42 + k * 7, yy - 1, 42 + k * 7 + 4, yy + 6, fill=c)
        self._t(d, 246, y, s.get("afm", "AF-C"), self.small)
        # file format and frame rate
        self._t(d, 620, y, s.get("codec", "4K"), self.small)
        self._t(d, 660, y, s.get("fmt", "24p"), self.small)
        # card
        d.polygon([(722 * S, 32 * S), (738 * S, 32 * S), (743 * S, 37 * S), (743 * S, 58 * S), (722 * S, 58 * S)],
                  outline=white, width=int(1.5 * S))
        self._t(d, 732.5, 52, "1", self.tiny, anchor="ms", shadow=False)
        self._t(d, 752, y, s.get("remain", "2h 14m"), self.small)
        # centre: STBY / REC
        if s.get("rec"):
            self._r(d, 406, 31, 474, 60, fill=(214, 28, 28, 255))
            self._t(d, 440, 54, "REC", self.med, anchor="ms", shadow=False)
        else:
            self._t(d, 440, 54, s.get("stby", "STBY"), self.med, anchor="ms")
        self._t(d, 490, 54, s.get("tc", "0:00:00"), self.med)
        # battery
        bx = 926
        self._r(d, bx, 35, bx + 48, 57, outline=white, w=1.5)
        self._r(d, bx + 48, 41, bx + 52, 51, fill=white)
        pct = s.get("battery", 84) / 100
        self._r(d, bx + 3, 38, bx + 3 + 42 * pct, 54, fill=white)
        self._t(d, bx - 8, 54, f'{s.get("battery", 84)}%', self.small, anchor="rs")
        # ---- settings shown while on standby
        y = 85
        self._r(d, 26, 66, 62, 91, outline=white, w=1.5)
        self._t(d, 44, 87, s.get("mode", "M"), self.med, anchor="ms")
        # film-strip tick marks on the mode box: "movie" mode
        for yy in (69, 75, 81):
            self._r(d, 28, yy, 30, yy + 3, fill=white)
        # AF-area bracket icon
        for x0, y0, dx, dy in ((86, 67, 1, 1), (112, 67, -1, 1), (86, 89, 1, -1), (112, 89, -1, -1)):
            self._l(d, [(x0, y0 + 6 * dy), (x0, y0), (x0 + 6 * dx, y0)], white)
        self._r(d, 97, 75, 101, 81, outline=white, w=1)
        self._t(d, 136, y, s.get("look", "ST"), self.small)
        # steady-shot: hand-ish glyph with waves
        cx = 202
        self._r(d, cx - 5, 69, cx + 5, 89, outline=white, w=1.4)
        for sgn in (-1, 1):
            d.arc([(cx + sgn * 9 - 8) * S, 69 * S, (cx + sgn * 9 + 8) * S, 89 * S],
                  start=-55 if sgn > 0 else 125, end=55 if sgn > 0 else 235, fill=white, width=int(1.3 * S))
        # ---- bottom bar
        y = 706
        fi = s.get("focus", "off")
        col = (255, 255, 255, 255)
        cx, cy = 42, 697
        if fi == "lock":
            d.ellipse([(cx - 9) * S, (cy - 9) * S, (cx + 9) * S, (cy + 9) * S], fill=(90, 230, 90, 255))
        elif fi == "hunt":
            if int(t * 6) % 2 == 0:
                d.ellipse([(cx - 9) * S, (cy - 9) * S, (cx + 9) * S, (cy + 9) * S], fill=col)
        else:
            d.ellipse([(cx - 9) * S, (cy - 9) * S, (cx + 9) * S, (cy + 9) * S], outline=(120, 120, 120, 255), width=int(1.2 * S))
        self._t(d, 86, y, s.get("shutter", "1/50"), self.big)
        self._t(d, 206, y, s.get("fnum", "F1.8"), self.big)
        # exposure scale
        x0, x1, ym = 322, 622, 696
        for i in range(13):
            xx = x0 + (x1 - x0) * i / 12
            h = 9 if i % 2 == 0 else 5
            self._l(d, [(xx, ym), (xx, ym - h)], white, 1.3)
        self._t(d, x0 - 14, ym + 1, "−", self.small, anchor="ms")
        self._t(d, x1 + 14, ym + 1, "+", self.small, anchor="ms")
        ev = float(np.clip(s.get("ev_meter", 0.0), -3, 3))
        xp = (x0 + x1) / 2 + ev / 3 * (x1 - x0) / 2
        d.polygon([(xp * S, (ym + 3) * S), ((xp - 6) * S, (ym + 13) * S), ((xp + 6) * S, (ym + 13) * S)], fill=white)
        self._t(d, 660, y, s.get("ev", "±0.0"), self.big)
        self._t(d, 772, y, "ISO", self.small)
        self._t(d, 808, y, s.get("iso", "6400"), self.big)
        self._t(d, 998, y, s.get("wb", "AWB"), self.med, anchor="rs")
        # time code, hours : minutes : seconds . frames
        self._t(d, 86, 735, s.get("timecode", "00:00:00.00"), self.small, dim)
        self._t(d, 250, 735, s.get("lens", "55mm"), self.small, dim)

    def _af(self, d, a):
        x, y, w, h = a["x"], a["y"], a["w"], a["h"]
        # a["x"], a["y"]: image coords 0..1 (y down), w/h: panel px
        px = x * W
        py = TOP + y * (BOT - TOP)
        al = int(255 * a.get("alpha", 1.0))
        state = a.get("state", "search")
        col = {"search": (255, 255, 255, al), "track": (255, 255, 255, al),
               "lock": (70, 240, 90, al), "fail": (255, 70, 60, al), "dim": (200, 200, 200, al)}[state]
        x0, x1, y0, y1 = px - w / 2, px + w / 2, py - h / 2, py + h / 2
        self._r(d, x0 - 1.5, y0 - 1.5, x1 + 1.5, y1 + 1.5, outline=(0, 0, 0, int(al * .55)), w=1)
        if a.get("corners"):
            c = min(w, h) * .28
            for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                self._l(d, [(cx, cy + c * dy), (cx, cy), (cx + c * dx, cy)], col, a.get("lw", 2.0))
        else:
            self._r(d, x0, y0, x1, y1, outline=col, w=a.get("lw", 2.0))
        if a.get("label"):
            self._t(d, x0, y0 - 7, a["label"], self.tiny, col)

    def _mag(self, d, m):
        # navigator: frame + magnified region, plus the factor
        nw, nh = 118, 66
        x1, y1 = 998, 656
        x0, y0 = x1 - nw, y1 - nh
        self._r(d, x0, y0, x1, y1, outline=(255, 255, 255, 230), fill=(0, 0, 0, 120), w=1.5)
        rx, ry, rz = m["x"], m["y"], m["zoom"]
        rw, rh = max(nw / rz, 2), max(nh / rz, 2)
        cx, cy = x0 + rx * nw, y0 + ry * nh
        self._r(d, cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2, fill=(255, 150, 40, 255))
        self._t(d, x0 - 12, y1 - 6, m["label"], self.big, anchor="rs")

    def _level(self, d, roll):
        cx, cy = W / 2, (TOP + BOT) / 2
        ang = roll
        ok = abs(ang) < .004
        col = (255, 255, 255, 255) if ok else (255, 255, 255, 215)
        c, s = np.cos(ang), np.sin(ang)
        for r0, r1 in ((60, 300), (-60, -300)):
            # the inner ends are turned up, or a line that thin across a picture is a scratch
            for w, ink in ((4.2, (0, 0, 0, 150)), (2.4, col)):
                self._l(d, [(cx + r0 * c, cy + r0 * s), (cx + r1 * c, cy + r1 * s)], ink, w)
                self._l(d, [(cx + r0 * c + s * 11, cy + r0 * s - c * 11), (cx + r0 * c - s * 11, cy + r0 * s + c * 11)], ink, w)

    def _hist(self, d, h, s):
        x1, y1 = 998, 660
        w, ht = 150, 72
        x0, y0 = x1 - w, y1 - ht
        self._r(d, x0, y0, x1, y1, outline=(255, 255, 255, 170), fill=(0, 0, 0, 100), w=1)
        h = np.asarray(h, "f4")
        h = h / max(h.max(), 1e-5)
        pts = [(x0, y1)]
        for i, v in enumerate(h):
            pts.append((x0 + w * i / (len(h) - 1), y1 - v * (ht - 3)))
        pts.append((x1, y1))
        d.polygon([(a * S, b * S) for a, b in pts], fill=(255, 255, 255, 215))

    def _menu(self, d, m):
        x0, y0 = 40, TOP + 26
        items = m["items"]
        row = 42
        self._r(d, x0, y0, x0 + 176, y0 + 18 + row * len(items), fill=(14, 14, 16, 225))
        self._t(d, x0 + 12, y0 + 2 - 8, m.get("title", ""), self.tiny, (220, 220, 220, 255))
        for i, it in enumerate(items):
            yy = y0 + 12 + i * row
            on = i == m["sel"]
            if on:
                self._r(d, x0 + 4, yy, x0 + 172, yy + row - 3, fill=(232, 120, 18, 255))
            f = self.zh if it.get("zh") else self.med
            self._t(d, x0 + 18, yy + 29, it["code"], f, shadow=False)
            if it.get("name"):
                self._t(d, x0 + 62, yy + 28, it["name"], self.tiny, (240, 240, 240, 255), shadow=False)

    def _banner(self, d, b):
        cx, cy = W / 2, (TOP + BOT) / 2
        self._r(d, cx - 280, cy - 44, cx + 280, cy + 44, fill=(10, 10, 10, 235), outline=(255, 255, 255, 180))
        self._t(d, cx, cy + 10, b, self.med, anchor="ms")
