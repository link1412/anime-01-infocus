"""The camera's display on the same tile hardware: quarter-panel bitmap, no anti-aliasing, palette only.

Same state keys and same layout as hud.py, drawn on a 256x192 grid and scaled x4: an item sits on the
block that holds hud.py's panel coordinate (panel px // 4). One thing it will not draw yet: the
recording and focus-locked states keep their looks for a later beat, so here the camera is always in
standby and every AF box is white.
"""
from __future__ import annotations
import numpy as np
from .worlds.pixel import BASE, N0, N3, N7, N9, SK2

W, H = 256, 192
TOP, BOT = 24, 168          # the image, in blocks


def _c(i):
    return (*(int(v) for v in BASE[i]), 255)


WHITE, DIM, OFF, AMBER, DARK = _c(N9), _c(SK2), _c(N3), _c(N7), _c(N0)


def _font(table, h):
    out = {}
    for line in table.strip().split("\n"):
        ch, rows = line.split(maxsplit=1)
        rows = rows.split()
        assert len(rows) == h, ch
        out[ch] = np.array([[c == "#" for c in r] for r in rows], bool)
    out[" "] = np.zeros((h, len(next(iter(out.values()))[0])), bool)
    return out


# 5x7, drawn for this display
BIG = _font("""
0 .###. #...# #..## #.#.# ##..# #...# .###.
1 ..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.
2 .###. #...# ....# ...#. ..#.. .#... #####
3 .###. #...# ....# ..##. ....# #...# .###.
4 ...#. ..##. .#.#. #..#. #####  ...#. ...#.
5 ##### #.... ####. ....# ....# #...# .###.
6 ..##. .#... #.... ####. #...# #...# .###.
7 ##### ....# ...#. ..#.. .#... .#... .#...
8 .###. #...# #...# .###. #...# #...# .###.
9 .###. #...# #...# .#### ....# ...#. .##..
A .###. #...# #...# ##### #...# #...# #...#
B ####. #...# #...# ####. #...# #...# ####.
C .###. #...# #.... #.... #.... #...# .###.
D ####. #...# #...# #...# #...# #...# ####.
E ##### #.... #.... ####. #.... #.... #####
F ##### #.... #.... ####. #.... #.... #....
G .###. #...# #.... #.### #...# #...# .###.
H #...# #...# #...# ##### #...# #...# #...#
I .###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.
J ..### ...#. ...#. ...#. ...#. #..#. .##..
K #...# #..#. #.#.. ##... #.#.. #..#. #...#
L #.... #.... #.... #.... #.... #.... #####
M #...# ##.## #.#.# #.#.# #...# #...# #...#
N #...# ##..# #.#.# #..## #...# #...# #...#
O .###. #...# #...# #...# #...# #...# .###.
P ####. #...# #...# ####. #.... #.... #....
Q .###. #...# #...# #...# #.#.# #..#. .##.#
R ####. #...# #...# ####. #.#.. #..#. #...#
S .###. #...# #.... .###. ....# #...# .###.
T ##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..
U #...# #...# #...# #...# #...# #...# .###.
V #...# #...# #...# #...# #...# .#.#. ..#..
W #...# #...# #...# #.#.# #.#.# ##.## #...#
X #...# #...# .#.#. ..#.. .#.#. #...# #...#
Y #...# #...# .#.#. ..#.. ..#.. ..#.. ..#..
Z ##### ....# ...#. ..#.. .#... #.... #####
. ..... ..... ..... ..... ..... .##.. .##..
: ..... .##.. .##.. ..... .##.. .##.. .....
/ ....# ....# ...#. ..#.. .#... #.... #....
% ##..# ##..# ...#. ..#.. .#... #..## #..##
+ ..... ..#.. ..#.. ##### ..#.. ..#.. .....
- ..... ..... ..... ##### ..... ..... .....
− ..... ..... ..... ##### ..... ..... .....
± ..#.. ..#.. ##### ..#.. ..#.. ..... #####
× ..... #...# .#.#. ..#.. .#.#. #...# .....
""".replace("  ", " "), 7)

# 3x5, for the small labels
SMALL = _font("""
0 ### #.# #.# #.# ###
1 .#. ##. .#. .#. ###
2 ### ..# ### #.. ###
3 ### ..# .## ..# ###
4 #.# #.# ### ..# ..#
5 ### #.. ### ..# ###
6 ### #.. ### #.# ###
7 ### ..# .#. .#. .#.
8 ### #.# ### #.# ###
9 ### #.# ### ..# ###
A .#. #.# ### #.# #.#
B ##. #.# ##. #.# ##.
C .## #.. #.. #.. .##
D ##. #.# #.# #.# ##.
E ### #.. ##. #.. ###
F ### #.. ##. #.. #..
G .## #.. #.# #.# .##
H #.# #.# ### #.# #.#
I # # # # #
J ..# ..# ..# #.# .#.
K #.# #.# ##. #.# #.#
L #.. #.. #.. #.. ###
M #.# ### ### #.# #.#
N ##. #.# #.# #.# #.#
O .#. #.# #.# #.# .#.
P ##. #.# ##. #.. #..
Q .#. #.# #.# ### .##
R ##. #.# ##. #.# #.#
S .## #.. .#. ..# ##.
T ### .#. .#. .#. .#.
U #.# #.# #.# #.# ###
V #.# #.# #.# #.# .#.
W #.# #.# #.# ### #.#
X #.# #.# .#. #.# #.#
Y #.# #.# .#. .#. .#.
Z ### ..# .#. #.. ###
. . . . . #
: . # . # .
/ ..# ..# .#. #.. #..
% #.# ..# .#. #.. #.#
+ ... .#. ### .#. ...
- ... ... ### ... ...
− ... ... ### ... ...
± .#. ### .#. ... ###
× ... #.# .#. #.# ...
""", 5)
# the one letter that has to fit inside the mode box
MODE = _font("""
M ##.## #.#.# #...#
""", 3)


def _icon(rows):
    return np.array([[c == "#" for c in r] for r in rows.split()], bool)


EAR_L = _icon("#.. #.. ###")
EAR_R = _icon("##. ##. #.#")
MODE_BOX = _icon("""
##########
##.......#
#........#
##.......#
#........#
##.......#
##########
""")
CARD = _icon("""
#####.
#....#
#.##.#
#..#.#
#..#.#
#....#
######
""")
AF_AREA = _icon("""
##...##
#.....#
.......
...#...
.......
#.....#
##...##
""")
STEADY = _icon("""
.#.###.#.
#..#.#..#
#..#.#..#
#..#.#..#
.#.###.#.
""")


def _glyph(font, ch):
    g = font.get(ch)
    if g is None:
        g = font.get(ch.upper())
    return g if g is not None else font[" "]


class HudPixel:
    def _text(self, a, x, base, s, font=BIG, col=WHITE, anchor="l"):
        """base is the bottom row of the glyphs; each glyph advances by its own width plus one."""
        glyphs = [_glyph(font, ch) for ch in s]
        w = sum(g.shape[1] + 1 for g in glyphs) - 1
        gx = x if anchor == "l" else (x - w if anchor == "r" else x - w // 2)
        for g in glyphs:
            y0 = base - g.shape[0] + 1
            if gx >= 0 and gx + g.shape[1] <= W and y0 >= 0 and y0 + g.shape[0] <= H:
                a[y0:y0 + g.shape[0], gx:gx + g.shape[1]][g] = col
            gx += g.shape[1] + 1

    @staticmethod
    def _put(a, icon, x, y, col=WHITE):
        a[y:y + icon.shape[0], x:x + icon.shape[1]][icon] = col

    @staticmethod
    def _fill(a, x0, y0, x1, y1, col):
        a[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = col

    def _box(self, a, x0, y0, x1, y1, col):
        """outline, one block thick; x1 and y1 inclusive."""
        self._fill(a, x0, y0, x1 + 1, y0 + 1, col)
        self._fill(a, x0, y1, x1 + 1, y1 + 1, col)
        self._fill(a, x0, y0, x0 + 1, y1 + 1, col)
        self._fill(a, x1, y0, x1 + 1, y1 + 1, col)

    def _line(self, a, x0, y0, x1, y1, col):
        x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            x = int(round(x0 + (x1 - x0) * i / n))
            y = int(round(y0 + (y1 - y0) * i / n))
            if 0 <= x < W and 0 <= y < H:
                a[y, x] = col

    def draw(self, s: dict) -> np.ndarray:
        a = np.zeros((H, W, 4), "u1")
        if s.get("visible", True):
            self._bars(a, s)
            if s.get("level") is not None:
                self._level(a, s["level"])
            if s.get("hist") is not None:
                self._hist(a, s["hist"])
            if s.get("mag"):
                self._mag(a, s["mag"])
            if s.get("menu"):
                self._menu(a, s["menu"])
        if s.get("af"):
            self._af(a, s["af"])
        if s.get("banner"):
            self._banner(a, s["banner"])
        return a.repeat(4, axis=0).repeat(4, axis=1)

    def _bars(self, a, s):
        t = s.get("t", 0.0)
        # ---- top row, baseline y=54: audio, focus mode, standby and time, format, card, battery
        y = 13
        al, ar = s.get("audio", (0.1, 0.1))
        for ear, lv, ly, by in ((EAR_L, al, 8, 9), (EAR_R, ar, 12, 12)):
            self._put(a, ear, 6, ly, DIM)
            for k in range(23):
                on = k / 23 < lv
                c = (AMBER if k > 19 else WHITE) if on else OFF
                self._fill(a, 11 + k * 2, by, 12 + k * 2, by + 2, c)
        self._text(a, 61, y, s.get("afm", "AF-C"), SMALL)
        self._text(a, 110, y, s.get("stby", "STBY"), SMALL, anchor="c")     # standby, always
        self._text(a, 122, y, s.get("tc", "0:00:00"), SMALL)
        self._text(a, 155, y, s.get("codec", "4K"), SMALL)
        self._text(a, 165, y, s.get("fmt", "24p"), SMALL)
        self._put(a, CARD, 180, 8)
        self._text(a, 188, y, s.get("remain", "2h 14m"), SMALL)
        bx = 231
        self._box(a, bx, 8, bx + 12, 14, WHITE)
        self._fill(a, bx + 13, 10, bx + 14, 13, WHITE)
        pct = s.get("battery", 84) / 100
        self._fill(a, bx + 2, 10, bx + 2 + int(round(9 * pct)), 13, WHITE)
        self._text(a, 229, y, f'{s.get("battery", 84)}%', SMALL, anchor="r")
        # ---- second row, baseline y=85: mode, AF area, look, steady shot
        y = 21
        self._put(a, MODE_BOX, 6, 16)
        self._text(a, 9, 20, str(s.get("mode", "M"))[:1], MODE)
        self._put(a, AF_AREA, 21, 16)
        self._text(a, 34, y, s.get("look", "ST"), SMALL)
        self._put(a, STEADY, 46, 17)
        # ---- bottom row, baseline y=706
        y = 176
        fi = s.get("focus", "off")
        ring = ((9, 172), (10, 172), (11, 172), (8, 173), (12, 173), (8, 174), (12, 174), (8, 175), (12, 175), (9, 176), (10, 176), (11, 176))
        if fi == "hunt" and int(t * 6) % 2 == 0:
            self._fill(a, 9, 173, 12, 176, WHITE)
            for xx, yy in ring:
                a[yy, xx] = WHITE
        else:                                                               # never focus-locked here
            for xx, yy in ring:
                a[yy, xx] = DIM
        self._text(a, 21, y, s.get("shutter", "1/50"), BIG)
        self._text(a, 51, y, s.get("fnum", "F1.8"), BIG)
        x0, ym = 82, 174                                                    # even ticks: 82..154, centre 118
        for i in range(13):
            xx = x0 + 6 * i
            self._fill(a, xx, ym - (2 if i % 2 == 0 else 1), xx + 1, ym + 1, WHITE)
        self._text(a, 78, ym + 1, "−", SMALL)
        self._text(a, 156, ym + 1, "+", SMALL)
        ev = float(np.clip(s.get("ev_meter", 0.0), -3, 3))
        xp = x0 + 36 + int(round(ev * 12))
        self._fill(a, xp, 176, xp + 1, 177, WHITE)
        self._fill(a, xp - 1, 177, xp + 2, 178, WHITE)
        self._text(a, 165, y, s.get("ev", "±0.0"), BIG)
        self._text(a, 192, y, "ISO", SMALL)
        self._text(a, 202, y, s.get("iso", "6400"), BIG)
        self._text(a, 250, y, s.get("wb", "AWB"), SMALL, anchor="r")
        # ---- lower line, baseline y=738
        self._text(a, 21, 184, s.get("timecode", "00:00:00.00"), SMALL, DIM)
        self._text(a, 62, 184, s.get("lens", "55mm"), SMALL, DIM)

    def _af(self, a, f):
        px, py = f["x"] * 1024, 96 + f["y"] * 576
        w, h = f["w"], f["h"]
        al = f.get("alpha", 1.0)
        if al < .33:
            return
        x0, x1 = int((px - w / 2) // 4), int((px + w / 2) // 4)
        y0, y1 = int((py - h / 2) // 4), int((py + h / 2) // 4)
        b = np.zeros((H, W, 4), "u1")
        if f.get("corners"):
            c = max(int(min(w, h) * .28 / 4), 2)
            for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                self._line(b, cx, cy, cx + c * dx, cy, WHITE)
                self._line(b, cx, cy, cx, cy + c * dy, WHITE)
        else:
            self._box(b, max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1), WHITE)
        if al < .66:                # fades by losing every other block, never by blending
            yy, xx = np.mgrid[0:H, 0:W]
            b[(xx + yy) % 2 == 0] = 0
        m = b[..., 3] > 0
        a[m] = b[m]
        if f.get("label"):
            self._text(a, x0, y0 - 2, f["label"], SMALL)

    def _mag(self, a, m):
        nw, nh = 30, 17
        x1, y1 = 249, 164
        x0, y0 = x1 - nw, y1 - nh
        self._fill(a, x0, y0, x1 + 1, y1 + 1, DARK)
        self._box(a, x0, y0, x1, y1, WHITE)
        rz = max(float(m["zoom"]), 1e-3)
        rw, rh = max(int(round(nw / rz)), 1), max(int(round(nh / rz)), 1)
        cx, cy = x0 + m["x"] * nw, y0 + m["y"] * nh
        rx0, ry0 = int(round(cx - rw / 2)), int(round(cy - rh / 2))
        rx0, ry0 = int(np.clip(rx0, x0 + 1, x1 - rw)), int(np.clip(ry0, y0 + 1, y1 - rh))
        self._fill(a, rx0, ry0, rx0 + rw, ry0 + rh, AMBER)
        self._text(a, x0 - 3, y1 - 2, m["label"], BIG, anchor="r")

    def _level(self, a, roll):
        cx, cy = W / 2, (TOP + BOT) / 2
        c, s = np.cos(roll), np.sin(roll)
        for r0, r1 in ((15, 75), (-75, -15)):
            self._line(a, cx + r0 * c, cy + r0 * s, cx + r1 * c, cy + r1 * s, WHITE)

    def _hist(self, a, h):
        x1, y1 = 249, 165
        w, ht = 37, 18
        x0, y0 = x1 - w, y1 - ht
        self._fill(a, x0, y0, x1 + 1, y1 + 1, DARK)
        self._box(a, x0, y0, x1, y1, WHITE)
        h = np.asarray(h, "f4")
        h = h / max(float(h.max()), 1e-5)
        for i in range(1, w):
            v = h[int(i / w * len(h))]
            n = int(round(v * (ht - 1)))
            if n > 0:
                self._fill(a, x0 + i, y1 - n, x0 + i + 1, y1, WHITE)

    def _menu(self, a, m):
        x0, y0 = 10, TOP + 6
        items = m["items"]
        row = 10
        self._fill(a, x0, y0, x0 + 44, y0 + 5 + row * len(items), DARK)
        self._text(a, x0 + 3, y0 - 1, m.get("title", ""), SMALL)
        for i, it in enumerate(items):
            yy = y0 + 3 + i * row
            if i == m.get("sel"):
                self._fill(a, x0 + 1, yy, x0 + 43, yy + row - 1, AMBER)
            self._text(a, x0 + 4, yy + 7, it["code"], BIG)
            if it.get("name"):
                self._text(a, x0 + 18, yy + 7, it["name"], SMALL)

    def _banner(self, a, b):
        cx, cy = W // 2, (TOP + BOT) // 2
        self._fill(a, cx - 70, cy - 11, cx + 71, cy + 12, DARK)
        self._box(a, cx - 70, cy - 11, cx + 70, cy + 11, WHITE)
        self._text(a, cx, cy + 3, b, BIG, anchor="c")
