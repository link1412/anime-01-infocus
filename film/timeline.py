"""The film as data: one take, every track a function of film time."""
from __future__ import annotations
import pathlib
import numpy as np
from .core import GL, Track, smooth
from .hand import Hands
from .worlds.flower import Flower
from .worlds.flower_arrival import T_LAND, T_OVER  # noqa: F401  (audio.py reads T_OVER from here)
from .worlds.words import Words
from .worlds.desk import Desk, Term, project, TAN
from .worlds.portrait import Portrait, EYE
from .worlds.sea import Sea, FIGURE_AZ, FIGURE_EL, FIGURE_H
from .worlds.ink import Ink
from .worlds.pixel import Pixel
from .hud_words import HudWords

WORLD = (1696, 954)
OVER = 1.06
ASPECT = WORLD[0] / WORLD[1]

# ---- the shape of the film ------------------------------------------------------------
T_SEA = 15.0                # the water bead has become the sea
SEA_HOLD = 2.0              # the side of the sea, once it has loaded: the shot people stop for
T_MENU = T_SEA + 9.6 + SEA_HOLD   # the camera menu
T_INK = T_MENU + 2.2        # impossible option chosen
T_WORDS = T_INK + 12.7      # the painting turns out to be written
T_STOP = T_WORDS + 7.2      # the sentence has landed; the magnifier goes into its last full stop
T_PIXEL = T_STOP + 1.35     # the full stop was a sun
T_DESK = T_PIXEL + 6.0      # back out: the sentence is on a screen
T_LIVE = T_DESK + 3.7       # the screen starts showing the camera
T_WHITE = T_LIVE + 5.0
T_BACK = T_WHITE + 2.1      # the viewfinder comes back
T_END = T_BACK + 21.54      # hard cut: on the second beep, one frame before anything could be seen
DURATION = T_END + 5.0      # the picture's clock, the hands and the sound are all cut to this: don't move it
TITLE_AT = T_END + .8       # black, then the name
TITLE_OUT = TITLE_AT + 2.8
CARD_AT = TITLE_OUT + .5    # then what it was
CARD_OUT = CARD_AT + 4.6
FILM_END = CARD_OUT + .5
E = T_BACK


def view_to_im(vx, vy):
    return .5 + vx / (2 * ASPECT / OVER), .5 - vy / (2 / OVER)


def im_to_view(ix, iy):
    return (ix - .5) * 2 * ASPECT / OVER, (.5 - iy) * 2 / OVER


def glass_to_im(g, gc, zoom):
    return view_to_im((g[0] - gc[0]) * zoom, (g[1] - gc[1]) * zoom)


# ---- what the film calls reality. the terminal in the film prints and edits this table.
REALITY_OFF = dict(tremor=E + 7.4, grain=E + 8.3, moire=E + 9.1, eyepiece=E + 9.9, room=E + 10.8)
RETURN = E + 21.4
T_CREEP = E + 14.6


def reality(name, t):
    off = REALITY_OFF[name]
    return float(1 - smooth(off, off + .10, t))


class AfActor:
    """Spring-driven box. Keys retarget it; under-damped springs give it its nerves."""
    RATE = 120

    def __init__(self, keys, duration):
        keys = sorted(keys, key=lambda k: k[0])
        n = int(duration * self.RATE) + 2
        self.a = np.zeros((n, 4))
        self.states = []
        self.alpha = np.ones(n)
        pos = np.array(keys[0][1][:4], float)
        vel = np.zeros(4)
        k_i, dt = 0, 1 / self.RATE
        target, hz, zeta, state, alpha = pos.copy(), 4.0, .55, "search", 1.0
        for i in range(n):
            t = i * dt
            while k_i < len(keys) and keys[k_i][0] <= t:
                _, (x, y, w, h), hz, zeta, state, alpha = keys[k_i]
                target = np.array([x, y, w, h], float)
                k_i += 1
            w_ = 2 * np.pi * hz
            acc = w_ * w_ * (target - pos) - 2 * zeta * w_ * vel
            vel += acc * dt
            pos += vel * dt
            self.a[i] = pos
            self.alpha[i] = alpha
            self.states.append(state)

    def at(self, t):
        i = int(np.clip(t * self.RATE, 0, len(self.a) - 1))
        return self.a[i], self.states[i], self.alpha[i]


def stepped(steps, t, z0=1.0):
    """focus magnifier: jumps in discrete, eased steps."""
    z = z0
    for t0, a, b in steps:
        if t >= t0:
            u = min((t - t0) / .36, 1.0)
            z = a * (b / a) ** (u * u * (3 - 2 * u))
    return z


# ---- the flower ------------------------------------------------------------------------
PHONE_SCALE = Track([(0, 1.0), (5.5, 1.0), (10.5, 1.06), (DURATION, 1.06)])
HV = (.5053, -.0888, .0602)   # the drop that hangs off the petal, as the camera sees it. Flower reports it
F_HERO = 3.393              # the flower, in diopters
F_SECOND = 1.86             # the other flower
F_SKIN, F_PICTURE = 3.481, 3.512    # the drop's surface; the picture it makes, which floats in front of it
FOCUS = Track([(0, 1.45), (1.25, 1.45), (1.62, F_HERO + .16), (1.95, F_HERO), (5.65, F_HERO), (6.3, F_SECOND), (7.3, F_SECOND),
               (7.95, F_HERO + .11), (8.3, F_HERO), (10.75, F_HERO), (11.3, F_SKIN), (11.95, F_SKIN), (12.45, F_PICTURE)])
MAG_STEPS = [(11.45, 1.0, 5.9), (12.75, 5.9, 11.7), (13.8, 11.7, 23.4), (14.5, 23.4, 46.8)]
# where the magnifier goes: the whole drop, its middle, the flower in it and the sea beside it, then the sea
GC_X = Track([(0, -.05), (8.3, -.05), (10.9, HV[0] - .145), (11.5, HV[0]), (12.75, HV[0]), (13.11, HV[0]), (13.8, HV[0]), (14.16, .5164), (14.5, .5164), (14.86, .5324)])
GC_Y = Track([(0, .02), (8.3, .02), (10.9, HV[1] - .04), (11.5, -.075), (12.75, -.075), (13.11, -.095), (13.8, -.095), (14.16, -.0938), (14.5, -.0938), (14.86, -.0992)])
STOP = Track([(12.75, 1.0), (13.05, .5), (13.8, .5), (14.1, .35)])     # it stops down as it goes in: a drop is not a good lens


def mag_zoom(t):
    return stepped(MAG_STEPS, t)


def mag_label(t):
    lab = None
    for t0, a, b in MAG_STEPS:
        if t >= t0:
            lab = f"×{b:.1f}"
    return lab


# ---- sea ---------------------------------------------------------------------------------
SEA_ROLL = Track([(T_SEA - .05, np.pi), (T_SEA + .2, np.pi), (T_SEA + 1.5, 0.0)])
T_WHIP = T_SEA + 5.6
SEA_END = (-.1087, .0408)     # sun 62% across, horizon 58% down: where the painting will take over
EDGE_AZ = .60
SEA_START = (-.22, -.05)       # near where the ray through the middle of the last magnified frame goes, and clear of the end of the surf
SEA_YAW = Track([(T_SEA + .05, SEA_START[0]), (T_SEA + 1.5, .0), (T_SEA + 2.5, .018), (T_SEA + 4.3, .10), (T_WHIP, .10), (T_WHIP + .34, EDGE_AZ + .36), (T_SEA + 7.5, EDGE_AZ + .36),
                 (T_SEA + 8.1 + SEA_HOLD, EDGE_AZ + .355), (T_SEA + 8.5 + SEA_HOLD, SEA_END[0]), (T_MENU, SEA_END[0])])
SEA_PITCH = Track([(T_SEA + .05, SEA_START[1]), (T_SEA + 1.5, -.095), (T_SEA + 8.1 + SEA_HOLD, -.095), (T_SEA + 8.5 + SEA_HOLD, SEA_END[1]), (T_MENU, SEA_END[1])])
SEA_EV = Track([(11.45, .15), (13.2, -.5), (T_SEA, -1.05), (T_MENU, -1.05)])
ROLL_LEVEL = Track([(T_SEA, np.pi), (T_SEA + 1.5, 0.0)])

# ---- ink ---------------------------------------------------------------------------------
INK_REVEAL = Track([(T_INK - .3, 0.0), (T_INK + 1.5, 1.0)])
INK_PAN = Track([(T_INK, 0.0), (T_INK + 5.0, .34), (T_INK + 6.3, .62), (T_INK + 6.9, 1.65), (T_INK + 8.2, 1.72), (T_INK + 12.7, 1.76)])
INK_BLUR = Track([(T_INK + 7.2, 0.0), (T_INK + 7.9, .85), (T_INK + 8.5, .78), (T_INK + 9.2, 0.0)])


def ink_zoom(t):
    """exponential, or it rushes through the only part worth seeing."""
    u = float(np.clip((t - (T_INK + 9.8)) / 2.7, 0, 1))
    return 60.0 ** (u * u * (3 - 2 * u))


# ---- words -------------------------------------------------------------------------------
W_FOCUS = Track([(0, 1.45), (1.8, 1.45), (2.4, F_HERO + .1), (2.65, F_HERO), (3.15, F_HERO), (3.65, .02), (4.25, .02), (4.85, F_HERO + .12), (5.15, F_HERO)])

# ---- the full stop -------------------------------------------------------------------------
ZMAX = 24.0
STOP_IN = [(T_STOP, 1.0, 5.9), (T_STOP + .45, 5.9, 11.7), (T_STOP + .9, 11.7, ZMAX)]
STOP_OUT = [(T_PIXEL + 4.6, ZMAX, 11.7), (T_PIXEL + 5.05, 11.7, 5.9), (T_PIXEL + 5.5, 5.9, 1.0)]
QUANT = 33.33       # monitor pixels per half glyph cell: the sentence is made of screen pixels


def stop_zoom(t):
    if t < T_PIXEL:
        return stepped(STOP_IN, t)
    return stepped(STOP_OUT, t, ZMAX)


def stop_label(t):
    z = stop_zoom(t)
    if z < 1.05:
        return None
    return "×5.9" if z < 8 else ("×11.7" if z < 17 else "×23.4")


# ---- desk --------------------------------------------------------------------------------
DIST = Track([(T_DESK, .62), (T_DESK + .35, .62), (T_DESK + 2.5, 1.03), (T_LIVE, 1.03), (T_WHITE, 1.16), (T_BACK, 1.0), (T_END, 1.0)])
OX = Track([(T_DESK, -.135), (T_DESK + .35, -.135), (T_DESK + 2.5, -.02), (T_LIVE, .01), (T_WHITE, .02), (T_BACK, .012)])
OY = Track([(T_DESK, .095), (T_DESK + .35, .095), (T_DESK + 2.5, .02), (T_LIVE, -.004), (T_BACK, -.006)])
TUN_YAW = Track([(T_LIVE, 0.0), (T_LIVE + 1.4, .022), (T_LIVE + 2.6, -.03), (T_LIVE + 3.6, .045), (T_WHITE, -.02)])
TUN_PITCH = Track([(T_LIVE, 0.0), (T_LIVE + 1.0, -.012), (T_LIVE + 2.4, .018), (T_LIVE + 3.6, -.02), (T_WHITE, .01)])
TUN_ROLL = Track([(T_LIVE, 0.0), (T_LIVE + 1.8, .02), (T_LIVE + 3.3, .10), (T_WHITE, .20)])
TUN_EV = Track([(T_DESK, .9), (T_LIVE - .1, .9), (T_LIVE + .2, -.48), (T_LIVE + 3.0, -.3), (T_LIVE + 4.4, 1.1), (T_WHITE, 4.5)])
END_EV = Track([(T_BACK, .9), (RETURN - 1.5, .9), (RETURN, 1.5)])
# the focus goes looking past the screen for whoever is there, and comes back, and goes again. it never gets there
END_FOCUS_RACK = Track([(E + 19.6, 0.0), (E + 19.95, .20), (E + 20.25, .05), (E + 20.7, .29), (E + 21.0, .18), (E + 21.4, .30), (T_END, .31)])
T_REC = E + 20.0            # nobody pressed it

GRAY = (205, 205, 205)
DIM = (120, 122, 128)
GREEN = (140, 205, 150)
ORANGE = (240, 165, 80)
SENTENCE = ["雨中的花。", "花瓣上的水珠。", "水珠里的海。"]
# what was asked for. nobody is typing this: there are no keys to be heard
ASKED = ["做一部动画：手机拍相机的取景器，取景器里还有世界。",
         "要让人惊讶。",
         "太长了。只留最好的镜头。"]
ASK_CPS, ASK_BEAT = 18, .6


def typed(s, t, t0, cps=26):
    if t < t0:
        return ""
    return s[:int((t - t0) * cps) + 1]


def after_sentence(t):
    lines = [(s, GRAY) for s in SENTENCE]
    cur = (len(lines), 0)
    if t > T_DESK + 2.2:
        cmd = typed("ffplay -f avfoundation -i 0", t, T_DESK + 2.2, 22)
        lines.append([("$ ", GREEN), (cmd, GRAY)])
        cur = (3, 2 + len(cmd))
    return lines, cur


TERM_ROWS = 15


def final_terminal(t):
    L = []
    t0 = T_BACK + 1.1
    for s in ASKED:
        txt = typed(s, t, t0, ASK_CPS)
        if txt:
            L.append([("> ", GREEN), (txt, GRAY)])
        t0 += len(s) / ASK_CPS + ASK_BEAT
    tcode = t0 - ASK_BEAT + .5
    if t > tcode:
        L.append(("", GRAY))
        L.append(("# film/timeline.py", DIM))
    code = [("tremor", "手抖"), ("grain", "噪点"), ("moire", "摩尔纹"), ("eyepiece", "目镜"), ("room", "房间的声音")]
    for i, (k, zh) in enumerate(code):
        if t > tcode + .22 + i * .09:
            off = t > REALITY_OFF[k]
            val = "0.0" if off else "1.0"
            pad = " " * (9 - len(k))
            L.append([(f"{k}{pad}= ", GRAY), (val, ORANGE if off else GRAY), (f"   # {zh}", DIM)])
    cursor = None
    tstill = REALITY_OFF["room"] + .4
    if t > tstill:
        L.append(("", GRAY))
        amp = 0.0
        if t > T_CREEP:
            amp = (.0003 + .0014 * float(smooth(T_CREEP, T_CREEP + 2.1, t))) * (1 + np.sin(t * 1.7) * .6)
        L.append([("motion  ", DIM), (f"{amp:.4f}", ORANGE if amp > 0 else GRAY)])
    if t > E + 16.0:
        L.append(("? 全部是 0。还在动。", ORANGE))
    if t > E + 16.7:
        q = "".join(("倒", "影"))   # the search below finds nothing, in source or in bytecode
        L.append([("$ ", GREEN), (typed(f'grep -rn "{q}" film/', t, E + 16.7, 34), GRAY)])
    if t > E + 17.45:
        L.append(("(无结果)", DIM))
    if t > E + 18.1:
        ask = typed("有人吗?", t, E + 18.1, 7)
        L.append([("> ", GREEN), (ask, (255, 255, 255))])
        cursor = (len(L) - 1, 2 + 2 * len(ask))
    if cursor is None:
        if int(t * 2.4) % 2 == 0 or t < tstill:
            cursor = (len(L), 0)
    # a terminal scrolls
    over = max(len(L) + 1 - TERM_ROWS, 0)
    if over:
        L = L[over:]
        if cursor is not None:
            cursor = (cursor[0] - over, cursor[1])
    return L, cursor


class Film:
    def __init__(self, gl: GL):
        self.gl = gl
        self.world = gl.target(WORLD)
        self.words = Words(gl, WORLD)
        self.desk = Desk(gl, WORLD)
        self.desk.portrait = Portrait(gl, WORLD)
        self.hud_words = HudWords()
        self.hands = Hands(DURATION, tremor=self._tremor, breath=self._breath)
        self.sea = Sea(gl, WORLD)
        self.flower = Flower(gl, WORLD, sea=self.sea)
        self.words.set_scene(self.flower.snapshot(4.0, (GC_X(0), GC_Y(0)), F_HERO, F_SECOND), self.flower.data["beads"], HV)
        self.ink = Ink(gl, WORLD)
        self.pixel = Pixel(gl, WORLD)
        self.sea_tmp = gl.target(WORLD)
        self.px_tmp = gl.target(WORLD, mips=True)
        self.px_bg = gl.target(WORLD)
        # the camera's meters read the real mix
        lev = pathlib.Path("out/audio/levels.npy")
        self.levels = np.load(lev) if lev.exists() else None
        if self.levels is None:
            print(f"[film] {lev} missing: the camera's level meters will be flat. Run `python -m film.audio` first.")
        # the type: the raindrops that become the sentence
        self.term = Term()
        cam0 = dict(dist=DIST(T_DESK), ox=OX(T_DESK), oy=OY(T_DESK))
        rng = np.random.default_rng(3)
        self.type_items = []
        k = 0
        for row, s in enumerate(SENTENCE):
            for ch, sx, sy in self.term.char_positions(row, s):
                vx, vy = project(sx, sy, cam0, ASPECT)
                em = (Term.FS / 2560 * .597 / DIST(T_DESK)) / TAN
                self.type_items.append((ch, vx, vy, em / .84, 5.0 + k * .06, vx + rng.uniform(-.5, .5)))
                k += 1
        # the last full stop of the sentence, whose ring sits low and to the left of its cell
        ch, vx, vy, size, _, _ = self.type_items[-1]
        assert ch == "。"
        self.stop_c = np.array([vx - 27.5 * size / 128, vy - 31.5 * size / 128])
        self.sun_im = tuple(self.pixel.sun_im())
        self.sun_view = np.array(im_to_view(*self.sun_im))
        self.af = AfActor(self._af_keys(), DURATION)
        # video feedback
        self.L = 8
        self.ring = [gl.target((1024, 768), dtype="f1", mips=True) for _ in range(self.L + 1)]
        self.ring_i = 0
        self.stills = None

    @staticmethod
    def _tremor(t):
        return reality("tremor", t)

    @staticmethod
    def _breath(t):
        off = REALITY_OFF["tremor"]
        base = 1 - smooth(off, off + .10, t)
        # stillness. then something the table does not contain
        return float(base + smooth(T_CREEP, T_CREEP + 1.7, t) * 1.9)

    # ------------------------------------------------------------------ the box
    def _stop_xf(self, z):
        """zoom z about the full stop's ring, which travels to where the sun will be."""
        g = np.log(max(z, 1.0)) / np.log(ZMAX)
        return float(z), self.stop_c, self.stop_c + (self.sun_view - self.stop_c) * g

    def _stop_im(self, z):
        _, _, p = self._stop_xf(z)
        return view_to_im(*p)

    def _af_keys(self):
        gc0 = (GC_X(4.0), GC_Y(4.0))
        K = []
        add = lambda t, x, y, w, h, hz=3.2, z=.55, st="search", al=1.0: K.append((t, (x, y, w, h), hz, z, st, al))
        add(0, .50, .50, 104, 76, st="dim", al=.75)
        add(1.25, .49, .47, 86, 66, hz=5, z=.4, st="search")
        add(1.72, .47, .43, 52, 52, hz=9, z=.38, st="search")
        for t in np.arange(2.1, 5.45, .10):
            tr = self.flower.runner_view(t)
            if tr:
                x, y = glass_to_im(tr[:2], gc0, 1.0)
                add(float(t), x, y, 52, 52, hz=5.5, z=.7, st="track")
        add(5.45, .40, .80, 52, 52, hz=2, z=.5, st="fail")
        add(5.72, .50, .52, 140, 104, hz=2.6, z=.6, st="dim", al=.8)
        add(7.5, .44, .42, 92, 92, hz=3, z=.5, st="search")
        add(8.05, .30, .36, 48, 48, hz=7, z=.42, st="search")
        add(8.32, .58, .62, 48, 48, hz=7, z=.42, st="search")
        # one lands. it knows what to do with that by now, and this time it doesn't lose it
        arr = self.flower.arrival
        for t in np.arange(T_LAND + .12, 11.36, .10):
            x, y = glass_to_im(arr.view(t)[:2], (GC_X(t), GC_Y(t)), 1.0)
            add(float(t), x, y, 52, 52, hz=5.5, z=.7, st="track")
        add(11.4, .5, .5, 150, 150, hz=2.2, z=.9, st="dim", al=0.0)
        # sea
        S = T_SEA
        add(S + 1.4, .5, .5, 92, 70, hz=4, z=.55, st="search", al=.9)
        add(S + 2.0, .63, .68, 46, 46, hz=7, z=.45, st="search")
        add(S + 2.9, .20, .29, 44, 44, hz=7, z=.42, st="track")
        add(S + 3.8, .29, .30, 44, 44, hz=4, z=.5, st="track")
        fx, fy = self._islet_im()
        add(S + 4.35, fx, fy, 30, 34, hz=2.0, z=.8, st="search")               # it pauses on something very small
        add(S + 5.3, fx, fy, 26, 30, hz=4, z=.8, st="search")
        add(T_WHIP, .14, .50, 60, 60, hz=2.5, z=.35, st="fail")
        add(S + 6.9, .5, .46, 110, 80, hz=3, z=.6, st="dim", al=.75)
        add(S + 8.2 + SEA_HOLD, .5, .5, 60, 60, hz=3, z=.6, st="search")
        add(T_MENU, .62, .5, 60, 60, hz=2, z=.8, st="dim", al=0.0)
        # ink
        I = T_INK
        add(I + 2.0, .55, .46, 90, 72, hz=2.5, z=.6, st="search", al=1.0)
        add(I + 3.6, .36, .60, 50, 50, hz=4, z=.6, st="search")
        for tb in np.arange(I + 4.6, I + 6.25, .1):                            # the boat, and whoever is standing in it
            bx_, by_ = self._boat_im(tb)
            add(float(tb), bx_, by_ - .012, 30, 34, hz=1.8 if tb < I + 4.9 else 4, z=.85, st="search")
        add(I + 6.3, .22, .5, 70, 70, hz=2.2, z=.35, st="fail")
        add(I + 7.2, .5, .5, 130, 100, hz=2.5, z=.6, st="dim", al=.85)
        add(I + 9.4, .5, .45, 52, 52, hz=3, z=.7, st="search", al=1.0)
        add(I + 10.1, .5, .5, 30, 30, hz=1.6, z=.9, st="search", al=.6)
        add(I + 11.2, .5, .5, 30, 30, st="dim", al=0.0)
        # words
        W = T_WORDS
        add(W + 2.0, .40, .52, 70, 56, hz=3, z=.6, st="search", al=1.0)
        add(W + 3.3, .30, .20, 40, 40, hz=6, z=.45, st="search")
        add(W + 3.8, .72, .78, 40, 40, hz=6, z=.45, st="search")
        hero_w = glass_to_im(HV[:2], (-.05, .02), 1.0)
        add(W + 4.3, hero_w[0], hero_w[1], 62, 62, hz=2.0, z=.8, st="search")
        add(W + 5.2, hero_w[0], hero_w[1] + .15, 62, 62, hz=1.4, z=.8, st="fail", al=.8)
        add(W + 6.0, .5, .95, 62, 62, hz=1.4, z=.8, st="dim", al=0.0)
        # the full stop: the box hugs it all the way in
        x, y = self._stop_im(1.0)
        add(T_STOP - .55, x, y, 26, 26, hz=2.4, z=.8, st="search", al=0.0)
        x, y = self._stop_im(5.9)
        add(T_STOP + .05, x, y, 56, 56, hz=4, z=.8, st="search", al=1.0)
        x, y = self._stop_im(11.7)
        add(T_STOP + .5, x, y, 104, 104, hz=4, z=.8, st="search")
        x, y = self._stop_im(ZMAX)
        add(T_STOP + .95, x, y, 196, 196, hz=4, z=.8, st="search")
        add(T_PIXEL + .5, x, y, 196, 196, hz=4, z=.8, st="search", al=0.0)
        # pixel: the box finds whoever is out on the jetty
        P = T_PIXEL
        add(P + 1.5, .5, .5, 92, 70, hz=4, z=.55, st="search", al=1.0)
        add(P + 2.2, .30, .42, 44, 44, hz=6, z=.5, st="search")
        for tb in np.arange(2.8, 4.4, .1):
            b = self.pixel.sprite_box(dict(t=P + tb, tau=float(tb), mosaic=1, reveal=1.0))
            add(float(P + tb), b[0] + b[2] / 2, b[1] + b[3] / 2, b[2] * 1024 + 4, b[3] * 576 + 4, hz=2.5 if tb < 3.1 else 6, z=.85, st="search")
        add(P + 4.4, .5, .74, 52, 84, hz=2.5, z=.85, st="search", al=0.0)
        # desk
        D = T_DESK
        add(D + .6, .42, .38, 96, 44, hz=3, z=.6, st="search", al=1.0)
        add(D + 1.5, .28, .29, 80, 40, hz=3, z=.6, st="search")
        add(D + 2.3, .30, .25, 60, 40, hz=3, z=.6, st="dim", al=.85)
        add(T_LIVE - .2, .5, .5, 60, 40, st="dim", al=0.0)
        # after the white: nothing. the box has nothing to look for, until it does
        self.eye_im = view_to_im(EYE[0] - .012, EYE[1] - .012)      # the glyph's ink, not its cell
        hx, hy = view_to_im(.015, .47)
        add(E + 18.65, .50, .50, 150, 120, hz=1.2, z=.9, st="dim", al=0.0)
        add(E + 18.85, .50, .46, 150, 120, hz=1.2, z=.9, st="dim", al=.8)
        add(E + 19.6, hx, hy, 190, 230, hz=.9, z=.9, st="search", al=1.0)         # first it finds someone
        add(E + 20.55, self.eye_im[0], self.eye_im[1], 44, 44, hz=1.5, z=.85, st="search", al=1.0)
        add(RETURN, self.eye_im[0], self.eye_im[1], 34, 34, hz=9, z=.5, st="lock", al=1.0)      # then one eye
        add(T_END, self.eye_im[0], self.eye_im[1], 34, 34, hz=9, z=.5, st="lock", al=1.0)
        return K

    @staticmethod
    def _boat_im(t):
        x = ((.640 - INK_PAN(t) * .6336 + 8) % 16) - 8 + .030
        return view_to_im(x, -.3895)

    @staticmethod
    def _islet_im():
        """the figure out on the rock, at the yaw and pitch the camera holds while the box pauses on it."""
        az, el = FIGURE_AZ, FIGURE_EL + FIGURE_H * .5
        yaw, pitch = SEA_YAW(T_SEA + 4.5), SEA_PITCH(T_SEA + 4.5)
        d = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        F = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
        R = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
        U = np.cross(F, R)
        return view_to_im(float(d @ R / (d @ F)) / .27, float(d @ U / (d @ F)) / .27)

    # ------------------------------------------------------------------ state of the film at t
    def state(self, t: float) -> dict:
        h = self.hands.at(t)
        a, ast, alp = self.af.at(t)
        af = dict(x=a[0], y=a[1], w=a[2], h=a[3], state=ast, alpha=alp, corners=ast == "dim")
        base_isp = dict(roll_small=h["roll"], ibis=(h["yaw"], h["pitch"]))
        audio = (.22, .2)
        if self.levels is not None:
            audio = tuple(self.levels[min(int(t * 30), len(self.levels) - 1)])
        hud = dict(t=t, af=af if alp > .02 else None, audio=audio, focus="off")
        ph = dict(scale=PHONE_SCALE(t))
        if t < T_SEA:
            return self._flower_state(t, h, hud, ph, base_isp)
        if t < T_INK:
            return self._sea_state(t, h, hud, ph, base_isp)
        if t < T_WORDS:
            return self._ink_state(t, h, hud, ph, base_isp)
        if t < T_PIXEL:
            return self._words_state(t, h, hud, ph, base_isp)
        if t < T_DESK:
            return self._pixel_state(t, h, hud, ph, base_isp)
        return self._desk_state(t, h, hud, ph, base_isp)

    def _flower_state(self, t, h, hud, ph, isp):
        fd, zoom, gc = FOCUS(t), mag_zoom(t), (GC_X(t), GC_Y(t))
        # AF-C: while it follows the drop, the focal plane goes with it
        r = self.flower.runner_view(min(t, self.flower.T_RUN[1]))
        if r is not None and 1.95 < t < 5.75:
            k = float(smooth(2.2, 3.0, t)) * (1 - float(smooth(5.45, 5.75, t)))
            fd = fd + (1 / r[2] - F_HERO) * k
        a = self.flower.arrival.focus(t)
        if a is not None:
            fd = fd + (a - fd) * float(smooth(T_LAND + .1, T_LAND + .5, t))
        isp.update(ev=SEA_EV(t), iso=.30 + .70 * float(smooth(13.0, 15.0, t)), zebra=0.0)
        lab = mag_label(t)
        hud.update(shutter="1/250" if t < 13.8 else "1/500", fnum="F2.8" if t < 12.75 else ("F5.6" if t < 13.8 else "F8"), iso="100",
                   focus="hunt" if hud["af"] and hud["af"]["state"] in ("fail", "dim") else "off",
                   ev_meter=-.3 + .12 * np.sin(t),
                   mag=dict(x=.5 + gc[0] * .13, y=.5 - gc[1] * .24, zoom=zoom, label=lab) if lab else None)
        # the panel's pixels beat against the phone's: on the open sky of the last magnification that is all you would see
        ph.update(defocus=.55 + 1.6 * (1 - min(t / 1.0, 1.0)) ** 2, band=.012, ghost=.012 + .05 * float(smooth(12, 15, t)), mask=.42 - .26 * float(smooth(13.8, 14.5, t)))
        return dict(world=dict(name="flower", t=t, fd=fd, zoom=zoom, gc=gc, stop=float(STOP(t)), morph=float(smooth(T_SEA - .44, T_SEA - .03, t)), end=(SEA_YAW(T_SEA), SEA_PITCH(T_SEA), .27)), isp=isp, hud=hud, ph=ph)

    def _sea_state(self, t, h, hud, ph, isp):
        yaw, pitch = SEA_YAW(t), SEA_PITCH(t)
        dy = (SEA_YAW(t + .01) - SEA_YAW(t - .01)) / .02
        crack = dict(t_arrive=T_WHIP + .22, edge_az=EDGE_AZ) if t < T_MENU else None
        # the side of the sea has its grain from how dark it is. facing the sun the phone is hardly amplifying anything, so the camera has to supply it
        side = T_SEA + 8.1 + SEA_HOLD
        sun = float(smooth(T_SEA, T_SEA + 2, t) * (1 - smooth(T_WHIP - .3, T_WHIP + .1, t)) + smooth(side, side + .4, t) * (1 - smooth(T_INK - .8, T_INK, t)))
        isp.update(ev=SEA_EV(t), iso=1.0 + .8 * sun, zebra=0.0, roll_small=h["roll"], rs=float(np.clip(dy * .012, -.08, .08)))
        # the phone opens the sunset up a little, as phones do. at the side of the sea it goes back to what it always did
        ph.update(ae_target=.10 + .015 * (1 - float(smooth(T_WHIP - .6, T_WHIP, t))), noise=1.0 + .3 * sun)
        hud.update(shutter="1/500", fnum="F8", iso="100", focus="hunt" if hud["af"] and hud["af"]["state"] in ("fail", "dim") else "off",
                   ev_meter=.4 + .1 * np.sin(t * .7), level=ROLL_LEVEL(t) + h["roll"] * 3 if t < T_SEA + 2.5 else None, look="ST")
        if t >= T_MENU:
            codes = ["ST", "PT", "NT", "VV", "VV2", "FL", "IN", "SH", "BW", "SE"]
            items = [dict(code=c) for c in codes] + [dict(code="墨", zh=True)]
            sel = int(np.clip((t - (T_MENU + .35)) / .15, 0, 10))
            hud["menu"] = dict(title="Creative Look", items=items, sel=sel)
            hud["look"] = codes[sel] if sel < 10 else "墨"
        ph.update(defocus=.5, ghost=.02, band=.018, veil=.012, mask=.16)
        return dict(world=dict(name="sea", t=t, yaw=yaw, pitch=pitch, roll=float(SEA_ROLL(t)), tan_half=.27, yaw_v=dy,
                               crack=crack), isp=isp, hud=hud, ph=ph)

    def _ink_state(self, t, h, hud, ph, isp):
        pan = INK_PAN(t)
        ps = (INK_PAN(t + .01) - INK_PAN(t - .01)) / .02
        isp.update(passthrough=1.0, ev=0.0, iso=0.0)
        hud.update(shutter="1/50", fnum="F1.8", iso="6400", look="墨")
        ph.update(defocus=.5, band=.012, ghost=.0, gain_min=.60, mask=.26)
        z = ink_zoom(t)
        if z > 1.05:
            hud["mag"] = dict(x=.5 + .1 / (2 * ASPECT), y=.5 - .22 / 2, zoom=z, label=f"x{z:.0f}")
        elif t > T_INK + 1.5:
            hud["hist"] = True
        return dict(world=dict(name="ink", t=t, pan=pan, pan_speed=ps, blur=INK_BLUR(t), zoom=z, zc=(.1, .22), hist=True,
                               reveal=float(INK_REVEAL(t))), isp=isp, hud=hud, ph=ph)

    def _words_state(self, t, h, hud, ph, isp):
        tau = t - T_WORDS
        fd = W_FOCUS(tau)
        isp.update(ev=.15, iso=.28)
        night = float(smooth(.15, .95, tau))
        ph.update(defocus=.5, band=.012, ghost=.007, gain_min=.60 - .25 * night, mask=.26 + .16 * night)
        hud.update(style="words", paper=1 - night)
        leave = float(smooth(5.1, 6.4, tau))
        z = stop_zoom(t)
        lab = stop_label(t)
        if lab:
            hud["mag"] = dict(x=.5, y=.5, zoom=z, label=lab)
        st = dict(name="words", t=t, t0=T_WORDS, tau=tau, fd=fd, gc=(-.05, .02),
                  night=night,
                  city=float(smooth(1.2, 2.4, tau)) * (1 - leave),
                  flower=float(smooth(1.2, 2.4, tau)),
                  rain=.6 * float(smooth(1.0, 2.2, tau)) * (1 - leave),
                  beads=.8 * float(smooth(1.2, 2.4, tau)) * (1 - leave),
                  fall=leave,
                  wall_scale=(.178 * .30 ** float(smooth(.55, 2.3, tau))) if tau < 2.6 else 0.0,
                  wall_fall=max(tau - .95, 0.0),
                  wall_glyph=.92 - .42 * float(smooth(.05, .8, tau)),
                  quant=QUANT if z > 1.01 else 0.0,
                  type_inst=self.words.type_instances(self.type_items, tau, self._stop_xf(z)))
        return dict(world=st, isp=isp, hud=hud, ph=ph)

    def _pixel_state(self, t, h, hud, ph, isp):
        tau = t - T_PIXEL
        z = stop_zoom(t)
        lab = stop_label(t)
        if lab:
            hud["mag"] = dict(x=.5, y=.5, zoom=z, label=lab)
        reveal = float(smooth(.35, 1.3, tau)) * (1 - float(smooth(4.6, 5.8, tau)))
        if tau < 3.0:
            mosaic = 3 if tau < .12 else (2 if tau < .24 else 1)
        else:
            mosaic = 3 if tau > 5.9 else (2 if tau > 5.78 else 1)
        isp.update(passthrough=1.0, ev=0.0, iso=0.0)
        hud.update(shutter="1/50", fnum="F1.8", iso="6400")
        ph.update(defocus=.5, band=.012, ghost=0.0, mask=.30)
        ph.update(Pixel.PH)
        return dict(world=dict(name="pixel", t=t, tau=tau, mosaic=mosaic, reveal=reveal, z=z), isp=isp, hud=hud, ph=ph)

    def _desk_state(self, t, h, hud, ph, isp):
        dist, ox, oy = DIST(t), OX(t), OY(t)
        st = dict(name="desk", t=t, dist=dist, ox=ox, oy=oy, moire=float(smooth(T_DESK, T_DESK + .5, t)), glow=1.0, bright=1.0)
        fd = 1 / dist
        ev = TUN_EV(t) if t < T_WHITE + .4 else END_EV(t)
        grain, moire, eye = reality("grain", t), reality("moire", t), reality("eyepiece", t)
        hud.update(shutter="1/50", fnum="F1.8", iso="3200", focus="hunt" if hud["af"] and hud["af"]["state"] in ("fail", "dim") else "off", ev_meter=.1)
        ph.update(defocus=.5, band=.016)
        if t < T_LIVE:
            lines, cur = after_sentence(t)
            st.update(lines=lines, cursor=cur if int(t * 2.4) % 2 == 0 or t > T_DESK + 2.2 else None)
            if T_DESK + 2.8 < t < T_DESK + 3.3:
                fd += .22 * np.sin((t - T_DESK - 2.8) * 16)
        elif t < T_WHITE + .4:
            st.update(yaw=TUN_YAW(t), pitch=TUN_PITCH(t), roll=TUN_ROLL(t), feed=True, moire=.55)
            hud["af"] = None
            ph.update(band=.03, veil=.06, ae_tau=.9, gain_max=2.0, gain_min=.22)
        elif t < T_BACK:
            # the viewfinder blacks out, the phone opens up in the dark and finds the eyecup
            u = float(smooth(T_WHITE + .4, T_WHITE + .55, t))
            st.update(lines=[], cursor=None, bright=0.0, glow=0.0, led=0.0, blackout=True)
            hud = dict(t=t, visible=False)
            ph.update(ae_tau=.42, gain_max=12.0, rubber=.07 + .23 * u, band=.0, defocus=.5 + 4.5 * float(np.sin(t * 5.1) ** 2) * u, ae_target=.04)
        else:
            lines, cur = final_terminal(t)
            if t >= E + 18.8 and int(t * 2.4) % 2:
                cur = None          # the question has been asked. the cursor waits
            st.update(lines=lines, cursor=cur)
            k = float(END_FOCUS_RACK(t))
            ease = k * k * (3 - 2 * k)
            fd = 1 / dist + (1 / (2 * dist) - 1 / dist) * ease
            if t < T_BACK + .9:
                fd += .34 * np.sin((t - T_BACK) * 9) * (1 - (t - T_BACK) / .9)
            st.update(refl=.03 + .33 * float(smooth(E + 17.7, E + 19.1, t)) + .64 * float(smooth(E + 19.1, E + 21.1, t)), win=.075,
                      breath=float(np.sin(t * 1.7)), moire=moire)
            hud["focus"] = "lock" if t >= RETURN else ("hunt" if t > E + 19.0 else "off")
            if t >= T_REC:
                hud["rec"] = True
                run = t - T_REC
                sec = int(run)
                hud["tc"] = f"0:00:{sec:02d}"
                hud["timecode"] = f"00:00:{sec:02d}.{int((run - sec) * 24):02d}"
            # the things the terminal switches off, it really does switch off
            ph.update(noise=grain, mask=.42 * moire, band=.016 * grain, dust=eye, k1=.055 * eye, ca=.0035 * eye, fc=2.1 * eye,
                      pupil_r=.86 + 12 * (1 - eye), veil=.022 * eye, ghost=.012 * eye, rubber=.07 * eye, defocus=.5 * eye,
                      sharp=.72 * grain, tnr=.62 * grain, flat=1.0 if eye < .5 else 0.0, psf=.25 + .75 * eye)
        isp.update(ev=ev, iso=.75 * (grain if t >= T_BACK else 1.0))
        st["fd"] = fd
        return dict(world=st, isp=isp, hud=hud, ph=ph)

    # ------------------------------------------------------------------ hud dispatch
    def draw_hud(self, s, default):
        h = s["hud"]
        name = s["world"]["name"]
        if name == "words":
            return self.hud_words.draw(h)
        if name == "ink":
            return self.ink.draw_hud(h)
        if name == "pixel":
            return self.pixel.draw_hud(h)
        return default.draw(h)

    # ------------------------------------------------------------------ pixels
    def _still_times(self):
        return [4.2, T_SEA + 3.0, T_INK + 4.0, T_WORDS + 3.7, T_PIXEL + 3.0]

    def _ensure_stills(self, optics):
        """the earlier worlds, as the camera's own panel showed them, for the rings of the tunnel."""
        if self.stills is not None:
            return
        self.stills = []
        blank = np.zeros((768, 1024, 4), "u1")
        for t in self._still_times():
            s = self.state(t)
            w = self.render_world(s, optics, 0)
            target = self.gl.target((1024, 768), dtype="f1", mips=True)
            optics.body(w, blank, s["isp"], 0, OVER, into=target)
            self.stills.append(target)

    def render_world(self, s: dict, optics=None, frame=0):
        w = dict(s["world"])
        name = w["name"]
        if name == "flower":
            self.flower.render(w, self.world)
        elif name == "sea":
            self.sea.render(w, self.world)
        elif name == "ink":
            if w["reveal"] < 1:
                self.sea.render(dict(t=w["t"], yaw=SEA_END[0], pitch=SEA_END[1], roll=0.0, tan_half=.27, yaw_v=0.0, drop=None, crack=None), self.sea_tmp)
                w["from"] = self.sea_tmp
            k = float(smooth(T_WORDS - .34, T_WORDS - .03, w["t"]))
            if k > 0:
                # the painting's characters hand over to the wall of the next world
                self.ink.render(w, self.sea_tmp)
                self.words.render(dict(t=w["t"], t0=T_WORDS, tau=0.0, fd=W_FOCUS(0.0), gc=(-.05, .02), night=0.0, city=0.0, rain=0.0, beads=0.0,
                                       wall_scale=.178, wall_fall=0.0, wall_glyph=.92), self.px_bg)
                self.gl.run("words_cross.frag", self.world, uInk=self.sea_tmp, uWords=self.px_bg, uK=k, uEV=.15)
            else:
                self.ink.render(w, self.world)
        elif name == "words":
            self.words.render(w, self.world)
        elif name == "pixel":
            self._render_pixel(w)
        elif name == "desk":
            if w.get("win") and self.desk.window is None:
                # the same night as the first shot, still out there behind whoever is holding the camera
                self.desk.make_window()
            if w.get("feed"):
                if optics is not None:
                    self._ensure_stills(optics)
                self.desk.render(w, self.world, feed=self.ring[(self.ring_i + 1) % len(self.ring)].tex)
            else:
                self.desk.render(w, self.world)
        return self.world

    def _render_pixel(self, w):
        z = w["z"]
        if z >= ZMAX - 1e-3:
            self.pixel.render(w, self.world)
            return
        # on the way out the whole frame shrinks around its sun, and the sentence is waiting around it
        self.pixel.render(w, self.px_tmp)
        self.px_tmp.gen()
        xf = self._stop_xf(z)
        self.words.render(dict(t=w["t"], t0=T_WORDS, tau=99.0, fd=1.10, gc=(-.05, .02), night=1.0, city=0.0, rain=0.0, beads=0.0, fall=1.0,
                               wall_scale=0.0, quant=QUANT, type_inst=self.words.type_instances(self.type_items, 99.0, xf)), self.px_bg)
        vs, vp = self.sun_view, xf[2]
        self.gl.run("words_shrink.frag", self.world, uBg=self.px_bg, uFg=self.px_tmp, uS=float(z / ZMAX), uEV=.15,
                    uSun=(.5 + vs[0] / (2 * ASPECT), .5 + vs[1] / 2), uPos=(.5 + vp[0] / (2 * ASPECT), .5 + vp[1] / 2))

    def after_body(self, s, optics, frame):
        """the camera's output goes back to the monitor, a few frames late."""
        w = s["world"]
        if w["name"] == "desk" and T_LIVE - .4 < w["t"] < T_WHITE + .4:
            isp = dict(s["isp"])
            if self.stills:
                # around the monitor, each ring of the tunnel keeps a different world
                isp["bg_disp"] = self.stills[(frame // self.L) % len(self.stills)]
            optics.body(self.world, None, isp, frame, OVER, into=self.ring[self.ring_i])
            self.ring_i = (self.ring_i + 1) % len(self.ring)
