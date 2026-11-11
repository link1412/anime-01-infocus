"""The same place again, drawn by hand on old tile hardware, grown out of a full stop.

Everything is indexed colour: a 243x137 frame of palette indices composed from parallax layers
that move in whole pixels, then looked up through 32 palette entries. Water and glitter move by
palette cycling, the entry and exit move palette entries in steps, and nothing is ever blended.

Block grid (fixed): 7 world px per native px. With wx, wy in world px and wy measured from the top,
    col = floor(wx / 7), row = floor((wy + 2) / 7)
so the sun's centre block (col 148, row 68) sits on im (.62, .50) within half a world px.

State: t, tau, mosaic, reveal, pan. `pull`, `pcx` and `pcy` are accepted and ignored.
"""
from __future__ import annotations
import numpy as np
import moderngl
from ..core import GL, Target

BLOCK = 7
GRID_Y0 = -2                # world y (from the top) where native row 0 starts
FR_W, FR_H = 243, 137       # 1696/7 and (954+2)/7, rounded up
LW = 512                    # width of every wrapping layer
TR = 255                    # transparent index
CROP_X, CROP_Y = 48, 27     # world px cropped on each side by the isp's OVER=1.06
IM_W, IM_H = 1600, 900      # world px that reach the panel image

SUN_X, SUN_Y = 148, 68      # centre block of the sun
SUN_D = 39
HORIZON = 79                # first row of water, im y = .582
DECK = 108                  # the row the person's feet are on

WHITE = (236, 232, 222)
BLACK = (8, 8, 10)
GREY = (111, 109, 105)      # the glyph's anti-aliased bit: 45% from black to white

# ---- palette ---------------------------------------------------------------------------------
SK0, SK1, SK2, SK3, SK4, SK5, SK6, SUN, SUNC, CLD = range(10)                    # sky
N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, B0, B1, B2 = range(10, 23)              # ground
RA0, RA1, RA2, RB0, RB1, RB2, GL0, GL1, GL2 = range(23, 32)                     # cycled water

BASE = np.array([
    (54, 52, 112), (84, 64, 128), (126, 76, 130), (174, 90, 116), (220, 114, 92),
    (246, 154, 86), (255, 194, 102), (255, 222, 136), (255, 246, 204), (66, 50, 102),
    (20, 12, 30), (38, 24, 58), (58, 36, 84), (84, 50, 104), (118, 66, 116), (160, 84, 116),
    (226, 122, 84), (246, 160, 80), (255, 206, 104), WHITE,
    (70, 36, 52), (112, 60, 60), (170, 96, 72),
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
], "i4")
RING_A = [N4, N4, N5, N5, N6, N5, N5, N4]
RING_B = [N2, N3, N3, N3, N4, N4, N3, N3]
RING_G = [N6, N7, N7, N8, SUNC, N8, N7, N7]
# when each entry starts to wake, on the reveal axis: outward from the sun.
#   sun .10, sky .14 to .42, sea and ridges .44 to .60, jetty and town .70 to .86
WAKE = np.zeros(32, "f4")
WAKE[[SUN, SUNC]] = .10
WAKE[[SK6, SK5, SK4, SK3, SK2, SK1, SK0, CLD]] = [.14, .18, .22, .27, .32, .37, .42, .40]
WAKE[[GL0, GL1, GL2, RA0, RA1, RA2, N6, N5, N4, N3, RB0, RB1, RB2, N2]] = \
    [.44, .44, .44, .46, .46, .46, .46, .50, .53, .56, .58, .58, .58, .60]
WAKE[[N8, N7, B2, B1, B0, N9, N1, N0]] = [.70, .72, .74, .76, .78, .80, .82, .86]
WAKE_DUR, WAKE_STEPS = .14, 4
RING_ONLY = .004            # below this the frame is exactly the ring
IRIS = .14                  # the ring's interior is filled by this reveal
CLIP = (.46, .60)           # the sea rises over the bottom of the sun, one row at a time

CH = {".": TR, "k": N0, "i": N1, "v": N2, "p": N3, "m": N4, "r": N5, "c": N6, "a": N7, "g": N8, "w": N9,
      "b": B0, "n": B1, "t": B2}


def spr(art: str, table=CH) -> np.ndarray:
    """ASCII art to indices. Indentation is ignored; short rows are padded with transparency."""
    rows = [r.strip() for r in art.split("\n") if r.strip()]
    w = max(len(r) for r in rows)
    return np.array([[table[c] for c in r.ljust(w, ".")] for r in rows], "u1")


def put(layer, s, x, y):
    """Paint a sprite at native coordinates, wrapping in x."""
    h, w = s.shape
    for j in range(h):
        if not 0 <= y + j < layer.shape[0]:
            continue
        cols = (x + np.arange(w)) % layer.shape[1]
        m = s[j] != TR
        layer[y + j, cols[m]] = s[j][m]


def outline(x0, y0, text):
    """A silhouette written as pixel runs: '/' climbs, '\\' descends, 'a' is a run of a columns
    then one row, 'a^b' a run of a columns then b rows. Returns {col: top row}."""
    x, y, d = x0, y0, 1
    tops = {}
    for tok in text.split():
        if tok == "/":
            d = -1
        elif tok == "\\":
            d = 1
        else:
            a, _, b = tok.partition("^")
            for _ in range(int(a)):
                tops[x] = y
                x += 1
            y += d * (int(b) if b else 1)
    return tops


def disc(r2, n=SUN_D):
    """Clean pixel disc in an n x n box, from a squared radius chosen by hand per size."""
    c = (n - 1) // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1]
    return xx * xx + yy * yy <= r2


# ---- the ring: the full stop of the previous world, then its pixel-art refinement ------------
RING3 = """
...+#####+...
..+########..
.+##+...+###.
+##.......##+
##+.......+##
##.........##
##.........##
##.........##
##+.......+##
+##.......##+
.###+...+###.
..#########..
...+#####+...
"""
# 2px bits; the centre bit is 3px wide so that 39 stays symmetric about the sun's centre block
RING2 = """
......+#####+......
....+#########+....
...#############...
..#####.....#####..
.+###.........###+.
.###...........###.
+###...........###+
###.............###
###.............###
###.............###
###.............###
###.............###
+###...........###+
.###...........###.
.+###.........###+.
..#####.....#####..
...#############...
....+#########+....
......+#####+......
"""
DISC_OUT, DISC_IN = 380.25, 185.0       # diameters 39 and 27
IRIS_R2 = (110.25, 56.25, 20.25, 2.25)  # diameters 21, 15, 9, 3: the fill steps inward


def _ring_rgb():
    """{mosaic: frame} for reveal=0. Exactly the ring on near-black."""
    out = {}
    tone = {"#": WHITE, "+": GREY}
    x0, y0 = SUN_X - 19, SUN_Y - 19
    for m in (3, 2):
        art = RING3 if m == 3 else RING2
        rows = art.strip("\n").split("\n")
        n = len(rows)
        if m == 3:
            edges = [3 * i for i in range(n + 1)]
        else:
            edges = [2 * i for i in range(10)] + [18 + 3 + 2 * i for i in range(10)]
        f = np.empty((FR_H, FR_W, 3), "u1")
        f[:] = BLACK
        for j in range(n):
            for i in range(n):
                c = rows[j][i]
                if c != ".":
                    f[y0 + edges[j]:y0 + edges[j + 1], x0 + edges[i]:x0 + edges[i + 1]] = tone[c]
        out[m] = f
    f = np.empty((FR_H, FR_W, 3), "u1")
    f[:] = BLACK
    ring = disc(DISC_OUT) & ~disc(DISC_IN)
    f[y0:y0 + SUN_D, x0:x0 + SUN_D][ring] = WHITE
    out[1] = f
    return out


# ---- the person ------------------------------------------------------------------------------
# k body, a sun on the near edge, C the camera, w the phone's one bright pixel. Feet on the last row.
_P = {**CH, "C": B0, "L": B2, "T": B1}
_HEAD = """
    .......kkk....
    ......kkkkk...
    ......kkkka...
    ......kkkka...
    .......kkk....
    ........kk....
"""
_LEGS_STAND = """
    .......kkkk...
    .......kk.kk..
    .......kk.kk..
    .......kk.kk..
    .......kk.kk..
    .......kkk.kkk
"""
WALK = [spr(a, _P) for a in (
    "..............\n" + _HEAD + """
    .....kkkkkk...
    ...kkk.kkka...
    ...kk..kkka...
    ..kk...kkkkkk.
    .CC....kkk.kw.
    .CC....kkk....
    ......kkkkk...
    ......kk.kk...
    .....kk...kk..
    .....kk...kk..
    ....kk.....kk.
    ....kkk....kkk
""", _HEAD + """
    .....kkkkkk...
    .....kk.kka...
    .....kk.kkk...
    .....kk.kkkkk.
    .....kk.kk.kw.
    ....CC..kkk...
    ....CC..kkk...
    .......kkkk...
    .......kk.kk..
    .......kk.kk..
    .......kk..kk.
    .......kk.....
    .......kkk....
""", "..............\n" + _HEAD + """
    .....kkkkkk...
    .......kkka...
    .......kkka...
    ......kkkkkkk.
    ......kkkk.kw.
    ......kCCk....
    ......kCCkk...
    ......kk.kk...
    .....kk...kk..
    .....kk...kk..
    ....kk.....kk.
    ....kkk....kkk
""", _HEAD + """
    .....kkkkkk...
    .....kk.kka...
    .....kk.kkk...
    .....kk.kkkkk.
    .....kk.kk.kw.
    ....CC..kkk...
    ....CC..kkk...
    .......kkkk...
    .......kk.kk..
    ......kk..kk..
    .....kk...kk..
    ..........kk..
    ..........kkk.
""")]
STOP = spr("..............\n" + _HEAD + """
    .....kkkkkk...
    ....kk.kkka...
    ....kk.kkka...
    ....kk.kkkkkk.
    ...CC..kkk.kw.
    ...CC..kkk....
""" + _LEGS_STAND, _P)
RAISE = [spr(a, _P) for a in ("..............\n" + _HEAD + """
    .....kkkkkkTT.
    ....kk.kkk.CC.
    ....kk.kkkk.k.
    ....kk.kkkakk.
    .......kkk..kw
    .......kkkkkkk
""" + _LEGS_STAND, """
    ..............
    .......kkk....
    ......kkkkk....
    ......kkkkTTTT.
    ......kkkkCCCCL
    .......kkkCCCC.
    ........kk..k..
    .....kkkkkkkk..
    ......kkkkk....
    ......kkkk..kw.
    ......kkkkkkkk.
    .......kkk..kk.
    .......kkk....
""" + _LEGS_STAND)]
PERSON_AX = 8               # the sprite column that sits on the person's x
PERSON_X0 = 92              # on the jetty layer, at tau=0
WALK_TICKS = 36             # stops at tau=3.0
RAISE_TICK = 43             # tau=3.6
STEP_TICKS = 3


def _bbox(sprites):
    boxes = []
    for s in sprites:
        ys, xs = np.nonzero(s != TR)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    b = np.array(boxes)
    return int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())


def deorphan(l):
    """No lonely pixels: anything with no twin among its eight neighbours joins its surroundings.
    Diagonal runs survive, and cycled water counts as one colour family per entry."""
    out = l.copy()
    h, w = l.shape
    for y in range(1, h - 1):
        for x in range(w):
            c = l[y, x]
            if c == TR:
                continue
            nb = [l[y + dy, (x + dx) % w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]
            if c not in nb:
                four = [l[y - 1, x], l[y + 1, x], l[y, (x - 1) % w], l[y, (x + 1) % w]]
                vals, n = np.unique(four, return_counts=True)
                out[y, x] = vals[n.argmax()]
    return out


# ---- far ridge: the big mountain of the painting, its neighbour, and the low range behind the sun
FAR = (-64, 78, r"""
    / 10 8 7 6 5 5 4 4 3 3 3 2 2 2 6
      3 2 2 1 1 1 1 1 2 5
      3 2 1 1 1 1 1 1 1^2 1^2 1^2 1^2 1^2 1^2 1 1 1 1 1 2 2 3 4
    \ 3 2 1 1 1 1 1^2 1^2 1^2 1^2 1 1 1 1 1 1 1 1 1 1 2 2 3 4 1 1 2
    / 3 1 1
    \ 1 1 1 1^2 1^2 1^2 1^2 1^2 1^2 1 1 1 1 1 1 1 2 2 2 3 4
""")
# painted facets: (colour, left edge, right edge), each edge a list of (col, row) corners
FACETS = [
    # the big mountain's lit face
    (N5, [(59, 24), (55, 32), (49, 44), (41, 60), (33, 78)], [(200, 24), (200, 46), (82, 46), (88, 78)]),
    # gullies down the lit face
    (N4, [(63, 34), (54, 60)], [(63, 34), (60, 48), (54, 60)]),
    (N4, [(72, 40), (66, 52), (61, 66)], [(72, 40), (70, 52), (61, 66)]),
    (N4, [(80, 50), (72, 72)], [(80, 50), (78, 62), (72, 72)]),
    # lit ribs on the shadow side
    (N5, [(42, 40), (35, 54)], [(42, 40), (41, 46), (35, 54)]),
    (N5, [(30, 50), (22, 66)], [(30, 50), (29, 56), (22, 66)]),
    # the neighbour's lit face
    (N5, [(99, 49), (96, 56), (92, 78)], [(200, 49), (200, 78)]),
    (N4, [(106, 60), (100, 76)], [(106, 60), (105, 68), (100, 76)]),
    # the low range behind the sun: its lit faces look left
    (N5, [(160, 40), (160, 78)], [(221, 62), (224, 70), (229, 78)]),
    (N5, [(238, 40), (238, 78)], [(260, 58), (263, 68), (268, 78)]),
    (N4, [(208, 68), (200, 78)], [(208, 68), (207, 72), (200, 78)]),
    (N4, [(192, 73), (186, 78)], [(192, 73), (191, 75), (186, 78)]),
]
FAR_R = (174, 78, r"""
    / 9 7 5 4 3 3 2 2 2 1 1 1 2 3 5
    \ 3 2 1 1 2 3 4
    / 4 2 2 1 1 1 1 1 2 3 4
    \ 3 2 2 1 1 1 1 1 1 2 2 2 3 3 3 4 5 6 8
""")
# near ridge: domes of trees, each hand-written as a little climb and fall
NEAR = (-40, 54, r"""
    / 1 2 3 \ 2 1 1 1 / 1 1 3 \ 2 1 1 / 1 2 2 \ 3 1 1 1 1
    / 1 2 4 \ 2 1 1 1 / 1 3 \ 2 1 1 1 1 / 1 2 2 \ 2 1 1 1
    / 1 1 3 \ 2 2 1 1 1 / 2 3 \ 2 1 1 1 / 1 2 3 \ 2 1 1 1 1
    / 1 2 2 \ 3 1 1 1 / 1 3 \ 2 1 1 1 1 / 1 2 3 \ 3 1 1 1 1
    / 2 3 \ 2 1 1 1 1 / 1 2 \ 2 1 1 1 1 / 1 3 \ 2 2 1 1 1
    / 2 2 \ 3 1 1 1 1 / 1 3 \ 3 2 1 1 1 2 2 3 4 5 8
""")
NEAR_R = (214, 78, r"""
    / 6 4 / 1 2 3 \ 2 1 / 1 1 2 \ 2 1 / 1 2 3 \ 2 1 / 1 2 4 \ 3 1 / 1 2 3 \ 2 1 1
    / 2 3 \ 3 1 1 1 / 2 2 \ 2 1 1 1 2 2 3 4 5 8
""")
# the hill behind the town, on the jetty layer
HILL = (-40, 71, r"""
    \ 8 6 / 1 2 3 \ 2 1 / 1 2 4 \ 2 1 1 / 1 2 3 \ 3 1 1 / 1 2 2 \ 2 1 1
    / 1 2 3 \ 2 1 1 1 / 1 3 \ 2 2 1 1 1 1 1 / 2 2 \ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 3 3 4 5
""")

# ---- town ------------------------------------------------------------------------------------
HOUSE_A = spr("""
    ......a......
    .....bba.....
    ....bbbba....
    ...bbbbbba...
    ..bbbbbbbba..
    .bbbbbbbbbba.
    bbbbbbbbbbbba
    .nnnnnnnnnnt.
    .nnnnggnnnnt.
    .nnnnggnnnnt.
    .nnnnnnnnnnt.
    .nnggnnnggnt.
    .nnggnnnggnt.
    .nnggnnnggnt.
    .nnnnnnnnnnt.
    .nnnnnbbnnnt.
    .nggnnbbnnnt.
    .nggnnbbnnnt.
""")
HOUSE_B = spr("""
    ....bbbbbbbbbbbaa...
    ...bbbbbbbbbbbbbaa..
    ..bbbbbbbbbbbbbbbaa.
    .bbbbbbbbbbbbbbbbbaa
    bbbbbbbbbbbbbbbbbbba
    .nnnnnnnnnnnnnnnnnt.
    .nnggnnnnggnnnnggnt.
    .nnggnnnnggnnnnggnt.
    .nnggnnnnggnnnnggnt.
    .nnnnnnnnnnnnnnnnnt.
    .nnnnnnnnnnnnnnnnnt.
    .nnbbnnnnggnnnnggnt.
    .nnbbnnnnggnnnnggnt.
    .nnbbnnnnnnnnnnnnnt.
""")
HOUSE_C = spr("""
    ..nb.........
    ..nb..a......
    ..nb.bba.....
    ...bbbbba....
    ...bbbbbba...
    ..bbbbbbbba..
    .bbbbbbbbbba.
    bbbbbbbbbbbba
    .nnnnnnnnnt..
    .nnggnnggnt..
    .nnggnnggnt..
    .nnggnnggnt..
    .nnnnnnnnnt..
    .nnnnnnnnnt..
    .nnggnnggnt..
    .nnggnnggnt..
    .nnggnnggnt..
    .nnnnnnnnnt..
    .nnnnnbbnnt..
    .nggnnbbnnt..
    .nggnnbbnnt..
""")
SHED = spr("""
    ...bbbbbbbba..
    ..bbbbbbbbbba.
    .bbbbbbbbbbbba
    .nnnnnnnnnnnt.
    .nnbbbbnnggnt.
    .nnbbbbnnggnt.
    .nnbbbbnnnnnt.
    .nnbbbbnnnnnt.
""")
BACK_A = spr("""
    ......n.....
    .....iin....
    ....iiiin...
    ...iiiiiin..
    ..iiiiiiiin.
    .bbbbbbbbbn.
    .bbggbbggbn.
    .bbggbbggbn.
    .bbbbbbbbbn.
    .bbbbbbggbn.
    .bbbbbbggbn.
    .bbbbbbbbbn.
""")
BACK_B = spr("""
    ...iiiiiiiiin...
    ..iiiiiiiiiiin..
    .iiiiiiiiiiiiin.
    .bbbbbbbbbbbbbn.
    .bbggbbbbggbbbn.
    .bbggbbbbggbbbn.
    .bbbbbbbbbbbbbn.
    .bbbbbbbbbbbbbn.
""")
TOWER = spr("""
    ....a....
    ....a....
    ...bba...
    ...bba...
    ..bbbba..
    ..bbbba..
    .bbbbbba.
    .bbbbbba.
    .nnnnnnt.
    .nniignt.
    .nniignt.
    .nniignt.
    .nnnnnnt.
    .nnnnnnt.
    .bbbbbbn.
    .bbbbbbn.
    .bbggbbn.
    .bbggbbn.
    .bbbbbbn.
    .bbbbbbn.
    .bbbbbbn.
    .bbbbbbn.
    .bbggbbn.
    .bbggbbn.
    .bbbbbbn.
    .bbbbbbn.
""")
LAMP_POST = spr("""
    ..iii..
    .......
    .......
    .......
    .......
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ...k...
    ..kkk..
    ..kkk..
""")
LAMP_ON = spr("""
    ..iii..
    .iaaai.
    .agwga.
    .agwga.
    .iaaai.
""")
LAMP_DIM = spr("""
    ..iii..
    .iaaai.
    .aggga.
    .agwga.
    .iaaai.
""")
SAILBOAT = spr("""
    ....n.......
    ....ra......
    ....rca.....
    ....rcca....
    ....rcccc...
    ....rcccca..
    ....n.......
    .bbbbbbbbbb.
    ..bbbbbbbb..
""".replace("b", "v").replace("n", "v"))
DINGHY = spr("""
    ...........k.
    ...........k.
    a..........k.
    taaaaaaaaaakk
    .bbbbbbbbbbk.
    ..kkkkkkkkk..
""")
ROCK_L = spr("""
    kkkkkkbb.......................................
    kkkkkkkkbb..........k..........................
    kkkkkkkkkkbb.......kk..k.......................
    kkkkkkkkkkkkbb......kkkk.......................
    kkkkkkkkkkkkkkbbb....kk........................
    kkkkkkkkkkkkkkkkkbbb.kk........................
    kkkkkkkkkkkkkkkkkkkkbbbb.......................
    kkkkkkkkkkkkkkkkkkkkkkkkbbb....................
    kkkkkkkkkkkkkkkkkkkkkkkkkkkbbb.................
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkbbb..............
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkbbbb..........
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkbbbb......
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkbbbbb.
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
    kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
""")
POST = spr("""
    .ii.
    kkii
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
    kkin
""")

# ---- clouds: digits are roles, palette-swapped by height like old hardware --------------------
_CLOUDS = ["""
    ...............1111111........................
    .........1111112222222111.......111...........
    ....11112222222222222222211111112221111.......
    .1112222222333333322222222222223333332221111..
    ...3333333344444433333332223333344443333333...
    .......3344...333344444433....3344....333.....
""", """
    .......111111.........
    ...1111222222111......
    .11222233333222211....
    ...33334444333333.....
""", """
    .....22222222222............2222222.....
    .3333333333333333333.....33333333333333.
""", """
    ..........1111................
    .......111222211..111.........
    ....1112222222221122211.......
    ..11222222333322222222211.....
    .1222333333333333332222221....
    ...33344443333344444333333....
    .....444.....33344....33......
""", """
    ...................11111.......................
    ..............11112222211......111111..........
    ...........1112222222222211..1112222211........
    .........11222222222222222211222222222211......
    ......1112222222233333222222222223333222211....
    ....11222222222333333333322222233333333222111..
    ..112222233333333344444333333333344443333322211
    ....3333333333444444..44443333344444443333333..
    .......333444444.........4444444...444433......
    ...........44...............44.................
""", """
    ....333333333333333333333...........
    333333333444444444444333333333333...
    ......3333333333333333........333333
"""]
_C_HIGH = {"1": CLD, "2": SK1, "3": SK2, "4": SK3, ".": TR}
_C_MID = {"1": SK1, "2": SK2, "3": SK3, "4": SK4, ".": TR}
_C_LOW = {"1": SK2, "2": SK3, "3": SK4, "4": SK5, ".": TR}
# art, table, x, y, ticks per pixel of drift
CLOUDS = [
    (4, _C_HIGH, 186, 8, 56), (0, _C_HIGH, 106, 15, 40), (1, _C_HIGH, 36, 8, 48),
    (5, _C_MID, 196, 36, 34), (1, _C_MID, 116, 31, 34), (2, _C_MID, 18, 44, 44),
    (2, _C_LOW, 186, 54, 36), (5, _C_LOW, 214, 62, 30),
]
CLOUD_WRAP = 340


class Art:
    """All the hand-placed material, built once."""

    def __init__(self):
        self.sky = self._sky()
        self.clouds = [(spr(_CLOUDS[a], tb), x, y, r) for a, tb, x, y, r in CLOUDS]
        self.sun, self.sun_rank = self._sun()
        self.far, self.far_top = self._far()
        self.near = self._near()
        self.sea = self._sea()
        self.glitter = self._glitter()
        self.land = self._land()
        self.fore = self._fore()
        self.ring = _ring_rgb()
        for r in self.ring.values():
            r.flags.writeable = False
        self.boxes = {"walk": _bbox(WALK), "stop": _bbox([STOP]), "raise": _bbox(RAISE)}

    # ------------------------------------------------------------------ sky
    def _sky(self):
        yy, xx = np.mgrid[0:FR_H, 0:FR_W]
        d = np.hypot((xx - SUN_X) / 3.2, yy - SUN_Y)
        edges = np.array([21, 29, 39, 50, 62, 75], "f4")        # SK6 outwards
        half = 2.0                                              # half height of the dither bands
        idx = np.full((FR_H, FR_W), SK0, "u1")
        bayer = np.array([[.125, .625], [.875, .375]])[yy % 2, xx % 2]
        for i, e in enumerate(edges):
            inner = SK6 - i
            u = np.clip((e + half - d) / (2 * half), 0, 1)      # 1 inside, 0 outside
            u = np.ceil(u * 4 - 1e-6) / 4                       # 25, 50, 75% only
            mask = (u >= 1) | ((u > 0) & (u > bayer))
            idx[(d < e + half) & (idx == SK0) & mask] = inner
        return idx

    # ------------------------------------------------------------------ sun
    def _sun(self):
        d = disc(DISC_OUT)
        rim = d & ~disc(DISC_OUT - 72)
        s = np.full((SUN_D, SUN_D), TR, "u1")
        s[d] = SUNC
        s[rim] = SUN
        rank = np.full((SUN_D, SUN_D), 99, "u1")
        rank[d] = 0
        for i, r2 in enumerate((DISC_IN,) + IRIS_R2):
            rank[disc(r2)] = i + 1
        return s, rank

    # ------------------------------------------------------------------ far ridge
    def _far(self):
        tops = {**outline(*FAR), **outline(*FAR_R)}
        top = np.full(LW, HORIZON, "i4")
        for x, y in tops.items():
            top[x % LW] = min(y, HORIZON)
        l = np.full((FR_H, LW), TR, "u1")
        for c in range(LW):
            l[top[c]:HORIZON, c] = N4
        for col, left, right in FACETS:
            xl, xr = np.array(left, "f8"), np.array(right, "f8")
            for y in range(int(min(xl[:, 1].min(), xr[:, 1].min())), HORIZON):
                a = int(round(np.interp(y, xl[:, 1], xl[:, 0])))
                b = int(round(np.interp(y, xr[:, 1], xr[:, 0])))
                if y < xl[:, 1].min() or y > xl[:, 1].max() or y > xr[:, 1].max():
                    continue
                for x in range(a, b):
                    if l[y, x % LW] != TR:
                        l[y, x % LW] = col
        self._rim(l, top, N5)
        return deorphan(l), top

    @staticmethod
    def _rim(l, top, col):
        """A one-pixel light edge on the slopes that face the sun."""
        lit = np.zeros(LW, bool)
        for c in range(LW):
            x = c if c < 300 else c - LW
            nb = top[(c + 1) % LW] if x < SUN_X else top[(c - 1) % LW]
            lit[c] = nb > top[c] and top[c] < HORIZON
        for c in range(LW):
            x = c if c < 300 else c - LW
            nb = top[(c + 1) % LW] if x < SUN_X else top[(c - 1) % LW]
            if lit[c] and (lit[(c - 1) % LW] or lit[(c + 1) % LW]):
                l[top[c]:min(nb, HORIZON), c] = col

    # ------------------------------------------------------------------ near ridge
    def _near(self):
        tops = {**outline(*NEAR), **outline(*NEAR_R)}
        top = np.full(LW, HORIZON + 1, "i4")
        for x, y in tops.items():
            top[x % LW] = min(y, HORIZON + 1)
        l = np.full((FR_H, LW), TR, "u1")
        for c in range(LW):
            l[top[c]:HORIZON, c] = N2
        self._rim(l, top, N3)
        # its reflection: long dark runs that break up with distance from the shore
        for k in range(6):
            # each row is the ridge's foot pulled in a little and shifted, like a calm surface
            shift = (0, 2, -2, 3, -1, 2)[k]
            for c in range(LW):
                if (HORIZON - top[(c + shift) % LW]) // 3 > k:
                    l[HORIZON + k, c] = N2
        return deorphan(l)

    # ------------------------------------------------------------------ sea
    def _sea(self):
        rng = np.random.default_rng(11)
        l = np.full((FR_H, LW), N2, "u1")
        bands = [(HORIZON, 83, N5), (83, 91, N4), (91, 105, N3), (105, FR_H, N2)]
        for y0, y1, c in bands:
            l[y0:y1] = c
        l[:HORIZON] = TR

        def dash(y, x, n, c):
            l[y, (x + np.arange(n)) % LW] = c

        # interlock each band edge with long dashes of the neighbour
        for y0, y1, c in bands[1:]:
            above = bands[[b[2] for b in bands].index(c) - 1][2]
            for y, gap, n in ((y0 - 2, (26, 54), (8, 18)), (y0 - 1, (10, 28), (10, 24)),
                              (y0, (10, 26), (10, 26)), (y0 + 1, (24, 50), (8, 18))):
                x = int(rng.integers(0, 30))
                while x < LW:
                    ln = int(rng.integers(*n))
                    dash(y, x, ln, c if y < y0 else above)
                    x += ln + int(rng.integers(*gap))
        # ripples: lens-shaped clusters that grow toward the viewer, cycled through the palette
        y = HORIZON + 1
        while y < FR_H:
            t = (y - HORIZON) / (FR_H - HORIZON)
            n0, n1 = 3 + int(14 * t), 7 + int(24 * t)
            g0, g1 = 8 + int(10 * t), 20 + int(26 * t)
            ids = (RA0, RA1, RA2) if y < 92 else (RB0, RB1, RB2)
            x = int(rng.integers(0, 40))
            while x < LW:
                ln = int(rng.integers(n0, n1))
                k = int(rng.integers(0, 3))
                dash(y, x, ln, ids[k])
                if t > .3 and ln >= 8 and y + 1 < FR_H:
                    dash(y + 1, x + ln // 4, ln // 2, ids[(k + 1) % 3])
                x += ln + int(rng.integers(g0, g1))
            y += 1 + int(t * 2.6) + (1 if t > .15 else 0)
        # the far ridge lies on the water just below the horizon
        for k in range(5):
            shift = (0, 3, -2, 2, -3)[k]
            for c in range(LW):
                if (HORIZON - self.far_top[(c + shift) % LW]) // 4 > k:
                    l[HORIZON + k, c] = N3
        return deorphan(l)

    # ------------------------------------------------------------------ the sun on the water
    def _glitter(self):
        rng = np.random.default_rng(5)
        l = np.full((FR_H, FR_W), TR, "u1")
        G = (GL0, GL1, GL2)
        for y in range(HORIZON, FR_H):
            t = (y - HORIZON) / (FR_H - HORIZON)
            hw = 15 + 16 * t
            k = y % 3
            if y < HORIZON + 7:
                w = int(16 - (y - HORIZON) * 1.4)
                l[y, SUN_X - w:SUN_X + w + 1] = G[k]
                continue
            # a warm pool under the column, then broken dashes that thin out with distance
            if y % 2 == 0 and t < .55:
                w = int(hw * (.62 + .18 * ((y // 2) % 3)))
                l[y, SUN_X - w:SUN_X + w + 1] = N6
            x = SUN_X - int(hw) + int(rng.integers(0, 4))
            while x < SUN_X + hw:
                edge = abs(x - SUN_X) / hw
                ln = int(rng.integers(3, 6 + int(10 * t) + 1))
                gap = int(rng.integers(1, 3 + int(5 * t) + int(5 * edge) + 1))
                if rng.random() > edge * edge * .75:
                    l[y, x:x + ln] = G[(k + int(edge * 3)) % 3]
                x += ln + gap
        return deorphan(l)

    # ------------------------------------------------------------------ town and jetty
    def _land(self):
        l = np.full((FR_H, LW), TR, "u1")
        # the hill
        top = np.full(LW, FR_H, "i4")
        for x, y in outline(*HILL).items():
            top[x % LW] = y
        for c in range(LW):
            if top[c] < 100:
                l[top[c]:100, c] = N1
        # back row, then front row
        put(l, TOWER, 36, 72)
        put(l, BACK_A, 16, 84)
        put(l, BACK_B, 46, 88)
        put(l, HOUSE_A, 6, 83)
        put(l, HOUSE_B, 19, 87)
        put(l, HOUSE_C, 37, 81)
        put(l, SHED, 50, 93)
        # quay
        for x in range(-6, 68):
            l[101, x % LW] = B1
            l[102:107, x % LW] = N1
        for x, y, n in ((2, 103, 5), (12, 105, 4), (20, 103, 6), (33, 104, 4), (44, 103, 5), (54, 105, 5), (62, 103, 3)):
            l[y, (x + np.arange(n)) % LW] = N0
        # jetty
        x0, x1 = 66, 196
        l[DECK - 1:DECK + 1, x0:x1] = B1
        l[DECK + 1, x0:x1] = B2
        l[DECK + 2, x0:x1] = B0
        l[DECK + 3, x0:x1] = N1
        l[DECK - 1:DECK + 1, x0:x1][:, ::4] = B0
        for i, px in enumerate(range(70, x1, 16)):
            l[DECK + 4:DECK + 15, px:px + 2] = N1
            l[DECK + 4:DECK + 15, px + 1] = B0
            for k in range(1, 5):
                l[DECK + 15 + k * 2, (px - 1 + (k % 2)):(px + 3 - (k % 2))] = N1
            if i % 2 == 0:
                for j in range(8):
                    l[DECK + 5 + j, px + 2 + j * 2:px + 4 + j * 2] = N1
        # a broken band of shadow on the water beneath the deck
        for k, (on, off, ph) in ((16, (13, 4, 0)), (17, (9, 8, 5)), (19, (11, 12, 9)), (22, (6, 19, 3)), (25, (4, 27, 15))):
            for x in range(x0, x1):
                if (x + ph) % (on + off) < on:
                    l[DECK + k, x] = N1
        for x, y, n in ((9, 107, 4), (11, 109, 2), (21, 107, 3), (29, 108, 3), (30, 110, 2), (38, 107, 4), (40, 109, 3), (41, 112, 2),
                        (47, 108, 2), (56, 107, 3), (57, 109, 2)):
            l[y, x:x + n] = N7
        for y, x, n in ((124, 94, 5), (126, 95, 4), (129, 95, 3), (131, 96, 2)):
            l[y, x:x + n] = N7
        put(l, LAMP_POST, 93, 84)
        l[100:102, 69:79] = N1
        l[102, 72:77] = N1
        put(l, DINGHY, 200, 113)
        l[119, 202:214] = N1
        l[120, 205:212] = N1
        return l

    def _fore(self):
        l = np.full((FR_H, LW), TR, "u1")
        l[FR_H - 18:FR_H, -70 % LW:] = N0
        l[FR_H - 18:FR_H, :8] = N0
        put(l, ROCK_L, 8, FR_H - 18)
        put(l, POST, 224, 112)
        put(l, POST, 236, 118)
        return l


ART = None


def art():
    global ART
    if ART is None:
        ART = Art()
    return ART


def _im(col, row):
    return (col * BLOCK - CROP_X) / IM_W, (GRID_Y0 + row * BLOCK - CROP_Y) / IM_H


_COLS = np.arange(FR_W)


def _stamp(f, s, x, y):
    x0, x1 = max(x, 0), min(x + s.shape[1], FR_W)
    if x1 <= x0:
        return
    src = s[:, x0 - x:x1 - x]
    m = src != TR
    f[y:y + s.shape[0], x0:x1][m] = src[m]


def _over(f, layer, off, y0=0, y1=FR_H):
    src = layer[y0:y1][:, (_COLS + off) % layer.shape[1]]
    m = src != TR
    f[y0:y1][m] = src[m]


class Pixel:
    PH = dict(gain_max=1.5, ae_target=.19)      # the phone's exposure that suits this world

    def __init__(self, gl: GL | None, size=(1696, 954)):
        self.gl = gl
        self.size = tuple(size)
        self.art = art()
        self.idx = np.zeros((FR_H, FR_W), "u1")
        self.tex = None
        self.hud = None

    # ------------------------------------------------------------------ pure functions of st
    @staticmethod
    def _ticks(st):
        tau = max(float(st.get("tau", st.get("t", 0.0))), 0.0)
        return int(tau * 12 + 1e-6), int(tau * 8 + 1e-6)

    @staticmethod
    def _pan(st, tick):
        p = st.get("pan")
        return int(np.floor(p)) if p is not None else max(tick - 16, 0) // 3

    @staticmethod
    def _mosaic(st):
        return int(np.clip(round(float(st.get("mosaic", 1))), 1, 3))

    def _palette(self, reveal, ph):
        pal = BASE.copy()
        for k in range(3):
            pal[RA0 + k] = BASE[RING_A[(ph + k * 3) % 8]]
            pal[RB0 + k] = BASE[RING_B[(ph + k * 3) % 8]]
            pal[GL0 + k] = BASE[RING_G[(ph + k * 3) % 8]]
        if reveal >= 1:
            return pal
        s = np.floor(np.clip((reveal - WAKE) / WAKE_DUR, 0, 1) * WAKE_STEPS) / WAKE_STEPS
        two = np.tile(np.array(BLACK, "i4"), (32, 1))
        two[[SUN, SUNC]] = WHITE
        col = two + ((pal - two) * s[:, None]).astype("i4")
        mid = (s > 0) & (s < 1)
        col[mid] = col[mid] // 8 * 8                # five bits per channel while fading
        return col

    def _person(self, tick):
        if tick < WALK_TICKS:
            return WALK[tick // STEP_TICKS % 4], "walk"
        if tick < RAISE_TICK:
            return STOP, "stop"
        if tick < RAISE_TICK + 3:
            return RAISE[0], "raise"
        return RAISE[1], "raise"

    def _sun_into(self, f, reveal, y0, y1):
        a = self.art
        level = 0 if reveal <= 0 else int(np.ceil(np.clip(reveal / IRIS, 0, 1) * 5))
        clip = HORIZON
        if reveal < CLIP[1]:
            clip = HORIZON + int(np.ceil(np.clip((CLIP[1] - reveal) / (CLIP[1] - CLIP[0]), 0, 1) * 9))
        y1 = min(y1, clip, SUN_Y + 20)
        y0 = max(y0, SUN_Y - 19)
        if y1 <= y0:
            return
        sy = y0 - (SUN_Y - 19)
        h = y1 - y0
        s = a.sun[sy:sy + h]
        r = a.sun_rank[sy:sy + h]
        m = r <= level
        f[y0:y1, SUN_X - 19:SUN_X + 20][m] = s[m]

    def _compose(self, st, reveal, tick, ph):
        a = self.art
        pan = self._pan(st, tick)
        f = self.idx
        f[:] = a.sky
        for s, x, y, rate in a.clouds:
            _stamp(f, s, (x - tick // rate - pan // 8) % CLOUD_WRAP - 30, y)
        self._sun_into(f, reveal, 0, HORIZON)
        _over(f, a.far, pan // 4, 0, HORIZON)
        # the water scrolls by rows: every band nearer the viewer moves faster
        for y0, y1, off in ((HORIZON, 87, pan // 4), (87, 97, pan // 2), (97, 109, 3 * pan // 4), (109, FR_H, pan)):
            f[y0:y1] = a.sea[y0:y1][:, (_COLS + off) % LW]
        _over(f, a.glitter, 0, HORIZON, FR_H)
        self._sun_into(f, reveal, HORIZON, FR_H)
        _over(f, a.near, pan // 2, 40, HORIZON + 8)
        # the far boat keeps to the far water
        _stamp(f, SAILBOAT, 206 - pan // 4 + tick // 30, 84)
        _over(f, a.land, pan, 50, FR_H)
        self._tidy(f, HORIZON, FR_H - 1)
        # lamp: one pixel of flicker
        _stamp(f, LAMP_DIM if (tick * 7) % 19 < 3 else LAMP_ON, 93 - pan, 84)
        # the person
        s, _ = self._person(tick)
        _stamp(f, s, PERSON_X0 + min(tick, WALK_TICKS) - PERSON_AX - pan, DECK + 1 - s.shape[0])
        _over(f, a.fore, 2 * pan, 90, FR_H)
        return f

    @staticmethod
    def _tidy(f, y0, y1):
        """Rows are scrolled at different rates, so a dash cut by a band edge can leave a lone pixel."""
        a = f[y0 - 1:y1 + 1]
        c = a[1:-1, 1:-1]
        same = (c == a[:-2, 1:-1]) | (c == a[2:, 1:-1]) | (c == a[1:-1, :-2]) | (c == a[1:-1, 2:])
        same |= (c == a[:-2, :-2]) | (c == a[:-2, 2:]) | (c == a[2:, :-2]) | (c == a[2:, 2:])
        lone = ~same
        c[lone] = a[1:-1, :-2][lone]

    def frame_rgb(self, st):
        """The whole native frame, 137x243x3. Includes mosaic and reveal."""
        reveal = float(np.clip(st.get("reveal", 1.0), 0, 1))
        m = self._mosaic(st)
        if reveal <= RING_ONLY:
            return self.art.ring[m]
        tick, ph = self._ticks(st)
        f = self._compose(st, reveal, tick, ph)
        if m > 1:
            xs = SUN_X - 1 + ((_COLS - SUN_X + 1) // m) * m + m // 2
            ys = SUN_Y - 1 + ((np.arange(FR_H) - SUN_Y + 1) // m) * m + m // 2
            f = f[np.ix_(np.clip(ys, 0, FR_H - 1), np.clip(xs, 0, FR_W - 1))]
        return self._palette(reveal, ph).astype("u1")[f]

    # ------------------------------------------------------------------ the rest of the contract
    def render(self, st: dict, out: Target):
        if self.tex is None:
            self.tex = self.gl.ctx.texture((FR_W, FR_H), 3, dtype="f1", alignment=1)
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.tex.repeat_x = self.tex.repeat_y = False
        self.tex.write(self.frame_rgb(st).tobytes())
        self.gl.run("pixel_comp.frag", out, uTex=self.tex, uBlock=float(BLOCK), uY0=float(GRID_Y0),
                    uHeight=float(self.size[1]))

    def draw_hud(self, state):
        if self.hud is None:
            from ..hud_pixel import HudPixel
            self.hud = HudPixel()
        return self.hud.draw(state)

    def sun_im(self):
        """Centre of the sun in image coordinates (0..1, y down)."""
        x, y = _im(SUN_X + .5, SUN_Y + .5)
        return x, y

    def sprite_box(self, st):
        """(x, y, w, h) of the person in image coordinates, y down; x, y is the top-left corner.
        The box holds for a whole pose group, so it does not jitter with the walk cycle."""
        tick, _ = self._ticks(st)
        pan = self._pan(st, tick)
        s, group = self._person(tick)
        x0, y0, x1, y1 = self.art.boxes[group]
        px = PERSON_X0 + min(tick, WALK_TICKS) - PERSON_AX - pan
        py = DECK + 1 - s.shape[0]
        a, b = _im(px + x0, py + y0)
        c, d = _im(px + x1, py + y1)
        return a, b, c - a, d - b

    def lookdev(self, t):
        tau = max(t, 0.0)
        mosaic = 3 if tau < .12 else 2 if tau < .24 else 1
        reveal = float(np.clip((tau - .35) / .95, 0, 1)) * (1 - float(np.clip((tau - 4.6) / 1.4, 0, 1)))
        st = dict(t=t, tau=tau, mosaic=mosaic, reveal=reveal)
        x, y, w, h = self.sprite_box(st)
        af = dict(x=x + w / 2, y=y + h / 2, w=w * 1024 + 12, h=h * 576 + 12, state="track", alpha=1.0)
        hud = dict(shutter="1/50", fnum="F1.8", iso="6400", focus="off", audio=(.22, .2), af=af if 1.3 < tau < 4.6 else None)
        if tau < .5 or tau > 5.0:           # the magnifier is still on as the full stop opens and closes
            hud["mag"] = dict(x=.5, y=.5, zoom=23.4 if tau < 5.05 or tau > 9 else 11.7, label="×23.4" if tau < 5.05 else "×11.7")
        return {"world": st, "isp": dict(passthrough=1.0, ev=0.0, iso=0.0),
                "hud": hud, "ph": dict(scale=1.06, defocus=.5, band=.012, ghost=0.0, **self.PH)}
