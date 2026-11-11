"""The same flower, made of what it is made of.

Every point of it is a character, and a defocused character is still an aperture disc,
so the world only reveals its material when focus is pulled.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
import moderngl
from PIL import Image, ImageDraw
from ..core import GL, Target, smooth
from .. import fonts
from .flower import K_COC, TAN

CHARS = "雨中的花。瓣上水珠里海云光浪沙石蕊茎风滩"
IDX = {c: i for i, c in enumerate(CHARS)}
CELL = 128
GRID = 5
# the wall is still in the painter's hand when it arrives
KAI = {"山": len(CHARS), "水": len(CHARS) + 1}


def make_atlas(gl: GL, name="Songti.ttc", index=6):
    size = CELL * GRID
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    f = fonts.font(name, int(CELL * .84), index)
    for i, c in enumerate(CHARS):
        x, y = (i % GRID) * CELL + CELL // 2, (i // GRID) * CELL + CELL // 2
        d.text((x, y), c, font=f, fill=255, anchor="mm")
    k = fonts.font("Kaiti.ttc", int(CELL * .92), 0)
    for c, i in KAI.items():
        x, y = (i % GRID) * CELL + CELL // 2, (i // GRID) * CELL + CELL // 2
        d.text((x, y), c, font=k, fill=255, anchor="mm", stroke_width=2, stroke_fill=255)
    a = np.flipud(np.asarray(im, "u1"))
    t = gl.ctx.texture((size, size), 1, a.tobytes(), dtype="f1")
    t.build_mipmaps()
    t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    t.repeat_x = t.repeat_y = False
    return t


def cell_of(ch):
    return IDX[ch]


# how far away each thing is, in the same diopters as the flower it copies
D_HERO, D_SECOND = 3.393, 1.86


def _grid(pitch, rng, aspect):
    """a hex lattice over the frame, a little unsteady: nobody typeset this."""
    ys = np.arange(-1.12, 1.12, pitch * .90)
    xs = []
    for i, y in enumerate(ys):
        x = np.arange(-aspect - .1 + (i % 2) * pitch * .5, aspect + .1, pitch)
        xs.append(np.column_stack([x, np.full(len(x), y)]))
    g = np.vstack(xs)
    return g + rng.normal(0, pitch * .035, g.shape)


class Words:
    def __init__(self, gl: GL, size):
        self.gl = gl
        self.size = tuple(size)
        self.aspect = size[0] / size[1]
        ctx = gl.ctx
        self.atlas = make_atlas(gl)
        self.prog = gl.prog("words_glyph.frag", "words_glyph.vert")
        self.vb = ctx.buffer(reserve=12 * 4 * 60000, dynamic=True)
        fmt = "2f 1f 1f 1f 3f 1f 1f/i"
        names = ["ipos", "isize", "icell", "icoc", "icol", "ifade", "icell2"]
        self.vao = gl.instanced(self.prog, fmt, names, self.vb)
        self.t_add = gl.target(size)
        self.t_mask = gl.target(size)
        self.far = self.near = self.other = self.drops = self.hero = None
        rng = np.random.default_rng(21)
        # falling 雨 at several depths
        n = 340
        self.rain = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(0, 1, n), rng.uniform(.018, .05, n),
                                     rng.choice([3.4, 2.4, 1.7, 1.1, .6, .28], n, p=[.05, .18, .27, .25, .16, .09]),
                                     rng.uniform(.35, 1, n), rng.uniform(.8, 1.3, n), rng.uniform(0, 1, n)])
        # the wall of 山 left over from the painting
        gx, gy = np.meshgrid(np.arange(-34, 35), np.arange(-22, 23))
        self.wall = np.column_stack([gx.ravel() + rng.normal(0, .08, gx.size), gy.ravel() + rng.normal(0, .08, gx.size),
                                     rng.random(gx.size), rng.random(gx.size)])
        self.wall_cells = np.where(rng.random(gx.size) < .12, KAI["水"], KAI["山"])

    def set_scene(self, snap, beads, hero_view):
        """write the picture down: a character wherever there was a point of it, carrying that point's light."""
        rng = np.random.default_rng(8)
        H, W = snap["hero"].shape[:2]
        ppu = H / 2
        asp = self.aspect
        gc = snap["gc"]

        def cells(pitch):
            g = _grid(pitch, rng, asp)
            ix = np.clip(((g[:, 0] / asp + 1) * .5 * W).astype("i4"), 0, W - 1)
            iy = np.clip(((g[:, 1] + 1) * .5 * H).astype("i4"), 0, H - 1)
            return g, ix, iy, max(int(round(pitch * ppu)), 1)

        def box(a, k):
            return ndi.uniform_filter(a, (k, k, 0) if a.ndim == 3 else (k, k))

        ha = snap["hero"][..., 3]
        sa = snap["second"][..., 3]

        def energy(L, pitch):
            return L * (pitch * ppu) ** 2

        # ---- the near flower
        pitch = .066
        g, ix, iy, k = cells(pitch)
        rgb = box(snap["hero"][..., :3], k)[iy, ix]
        a = box(ha, k)[iy, ix]
        keep = a > .16
        g, rgb, a = g[keep], rgb[keep], a[keep]
        hue = rgb[:, 1] / np.maximum(rgb[:, 0], 1e-4)
        ch = np.where(hue > 1.0, cell_of("茎"), np.where(hue > .62, cell_of("蕊"), cell_of("花")))
        self.near = np.column_stack([g, np.full(len(g), pitch * .98), ch, energy(rgb, pitch), rng.random(len(g))]).astype("f4")
        # ---- the other one
        pitch = .058
        g, ix, iy, k = cells(pitch)
        rgb = box(snap["second"][..., :3], k)[iy, ix]
        a = box(sa, k)[iy, ix]
        over = 1 - box(ha, k)[iy, ix]
        keep = (a > .16) & (over > .1)
        g, rgb, a, over = g[keep], rgb[keep], a[keep], over[keep]
        hue = rgb[:, 1] / np.maximum(rgb[:, 0], 1e-4)
        ch = np.where(hue > 1.0, cell_of("茎"), cell_of("花"))
        self.other = np.column_stack([g, np.full(len(g), pitch * .98), ch, energy(rgb, pitch) * over[:, None], rng.random(len(g))]).astype("f4")
        # ---- what's behind them: sky, sea, sand
        pitch = .074
        g, ix, iy, k = cells(pitch)
        rgb = box(snap["far"], k)[iy, ix]
        vis = (1 - .96 * box(ha, k)[iy, ix]) * (1 - .96 * box(sa, k)[iy, ix])
        hz = float(np.tan(.10 - np.arctan(gc[1] * TAN)) / TAN)      # the horizon, where the flower's camera had it
        lum = rgb.mean(-1)
        y = g[:, 1]
        u = rng.random(len(g))
        ch = np.full(len(g), cell_of("沙"))
        ch[u < .07] = cell_of("滩")
        ch = np.where(lum < np.percentile(lum, 6), cell_of("石"), ch)
        sea = y > hz - .40
        ch = np.where(sea, np.where(u < .34, cell_of("浪"), cell_of("海")), ch)
        sky = y > hz
        ch = np.where(sky, np.where(u < .12, cell_of("风"), cell_of("云")), ch)
        ch = np.where(lum > np.percentile(lum, 97), cell_of("光"), ch)
        self.far = np.column_stack([g, np.full(len(g), pitch * .98), ch, energy(rgb, pitch) * vis[:, None], rng.random(len(g))]).astype("f4")
        # ---- water: every drop that sits on it is 珠, and the one that hangs is the sea
        z = beads[:, 2]
        bx, by = beads[:, 0] / z / TAN - gc[0], beads[:, 1] / z / TAN - gc[1]
        R = beads[:, 12] / z / TAN
        hv = np.array(hero_view[:2]) - np.array(gc)
        keep = np.hypot(bx - hv[0], by - hv[1]) > hero_view[2] * 1.5
        self.drops = np.column_stack([bx, by, np.maximum(R * 2.7, .026), z, rng.random(len(z))])[keep].astype("f4")
        self.hero = (float(hv[0]), float(hv[1]), float(hero_view[2] * 2.6))
        print(f"[words] {len(self.near)} + {len(self.other)} + {len(self.far)} characters, {len(self.drops)} drops", flush=True)

    def _area(self, size):
        """energy that makes a sharp glyph of this size come out at radiance 1."""
        return 4 * (np.asarray(size) * .5 * self.size[1] / 2) ** 2 * .30

    # ------------------------------------------------------------------
    def _draw(self, inst, target, blend=True, zoom=1.0, gain=1.0, quant=0.0):
        if len(inst) == 0:
            return
        inst = inst[:60000].astype("f4")
        self.vb.orphan()
        self.vb.write(inst.tobytes())
        gl, ctx = self.gl, self.gl.ctx
        gl.bind(self.prog, dict(uAtlas=self.atlas, uAspect=self.aspect, uPxPerUnit=self.size[1] / 2, uGrid=float(GRID), uGain=gain, uQuant=quant, uFlying=0.0))
        target.fbo.use()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(inst))
        ctx.disable(moderngl.BLEND)

    def render(self, st: dict, out: Target):
        gl = self.gl
        t = st["t"]
        fd, zoom = st["fd"], st.get("zoom", 1.0)
        gc = st.get("gc", (0.0, 0.0))
        night = st.get("night", 1.0)
        city_a = st.get("city", 1.0)
        rain_a = st.get("rain", 1.0)
        beads_a = st.get("beads", 1.0)
        fall = st.get("fall", 0.0)      # everything lets go, for the typesetting
        self.t_add.clear()
        self.t_mask.clear()
        # ---- far: the beach
        if city_a > .002 and self.far is not None:
            c = self.far
            n = len(c)
            coc = K_COC * abs(fd) * zoom * (1 + .14 * (c[:, 7] - .5))
            w = smooth(.55, 1.5, coc / (c[:, 2] * zoom * .5))
            inst = np.column_stack([c[:, 0] * zoom, c[:, 1] * zoom, c[:, 2] * zoom, c[:, 3], coc, c[:, 4:7] * city_a * (.5 + .5 * w)[:, None], np.ones(n), c[:, 3]])
            self._draw(inst, self.t_add)
        # ---- the flowers. when it is time they let go, character by character, and each one is rain by then
        flower_a = st.get("flower", 0.0)
        if flower_a > .002 and self.near is not None:
            for c, d in ((self.other, D_SECOND), (self.near, D_HERO)):
                n = len(c)
                fa = np.maximum(fall * 1.5 - c[:, 7] * .5, 0.0)
                yy = c[:, 1] - fa * fa * (1.3 + c[:, 7] * 1.4)
                cell = np.where(fa > .10, cell_of("雨"), c[:, 3])
                fade = np.clip(1.08 - fa * .55, 0, 1)
                # no two discs quite the same size, or their edges line up into a moire of their own
                coc = K_COC * abs(fd - d) * zoom * (1 + .14 * (c[:, 7] - .5))
                # a blurred character has only its light to give; a sharp one is a thin stroke, and if every
                # stroke clipped there would be no flower left, only a block of type
                w = smooth(.55, 1.5, coc / (c[:, 2] * zoom * .5))
                gain = (.38 + .62 * w)[:, None]
                inst = np.column_stack([c[:, 0] * zoom, yy * zoom, c[:, 2] * zoom, cell, coc, c[:, 4:7] * flower_a * gain, fade, cell])
                self._draw(inst, self.t_add)
        # ---- mid-air: rain
        if rain_a > .002:
            r = self.rain
            d = r[:, 3]
            speed = (.55 + d * .4) * r[:, 5]
            yy = 1.25 - np.mod(r[:, 1] * 2.6 + t * speed, 2.6)
            k = d / 3.4
            cc = K_COC * np.abs(fd - d) * zoom
            col = np.column_stack([.62 * r[:, 4], .74 * r[:, 4], 1.0 * r[:, 4]]) * (self._area(r[:, 2]) * .9 * rain_a)[:, None]
            inst = np.column_stack([(r[:, 0] - gc[0] * k) * zoom, (yy - gc[1] * k) * zoom, r[:, 2] * zoom * (1 + (d - .8) * .08),
                                    np.full(len(r), cell_of("雨")), cc, col, np.ones(len(r)), np.full(len(r), cell_of("雨"))])
            self._draw(inst, self.t_add)
        # ---- water: the drops on it are 珠, the drop that isn't like the others is 海
        if beads_a > .002 and self.drops is not None:
            a = self.drops
            n = len(a)
            coc = K_COC * np.abs(fd - 1 / a[:, 3]) * zoom
            yy = a[:, 1] - fall * fall * (1.4 + a[:, 4] * 1.6)
            b = a[:, 4]
            col = np.column_stack([.92 + .08 * b, .90 + .06 * b, .86 + .1 * b]) * ((.55 + .6 * b) * beads_a * self._area(a[:, 2]))[:, None]
            inst = np.column_stack([a[:, 0] * zoom, yy * zoom, a[:, 2] * zoom, np.full(n, cell_of("珠")), coc, col, np.ones(n), np.full(n, cell_of("珠"))])
            hx, hy, hs = self.hero
            hero = np.array([[hx * zoom, (hy - fall * fall * 2) * zoom, hs * zoom, cell_of("海"), K_COC * abs(fd - D_HERO) * zoom,
                              *(np.array([.72, .86, 1.0]) * 1.5 * beads_a * self._area(np.array(hs))), 1.0, cell_of("海")]])
            self._draw(np.vstack([inst, hero]), self.t_add)
        # ---- the wall of characters that came out of the painting
        wall_scale = st.get("wall_scale", 0.0)
        if wall_scale > 0:
            w = self.wall
            delay = w[:, 2] * .9 + w[:, 3] * .35
            fa = np.maximum(st.get("wall_fall", 0.0) - delay, 0.0)
            px = w[:, 0] * wall_scale
            py = w[:, 1] * wall_scale - fa * fa * 2.6
            fade = np.clip(1 - fa * .9, 0, 1)
            s = wall_scale * st.get("wall_glyph", .5)
            cell2 = np.where(fa > .05, cell_of("雨"), self.wall_cells)
            inst = np.column_stack([px, py, np.full(len(w), s * .94), self.wall_cells, np.zeros(len(w)),
                                    np.tile(self._area(np.array(s * .94))[..., None], (len(w), 3)), fade, cell2])
            self._draw(inst, self.t_mask)
        # ---- type: raindrops that stop falling and become a sentence
        type_inst = st.get("type_inst")
        if type_inst is not None and len(type_inst):
            self._draw(type_inst, self.t_add, quant=st.get("quant", 0.0))
        gl.run("words_comp.frag", out, uAdd=self.t_add, uMask=self.t_mask, uNight=night, uHasMask=1.0 if wall_scale > 0 else 0.0,
               uTime=t, uAspect=self.aspect)

    def type_instances(self, items, tau, xf=None):
        """items: (char, x, y, size, t0, x0): a raindrop that slows and lands as a character.
        xf: (zoom, centre, position), the magnifier going into the sentence."""
        out = []
        white = 1.02          # through this world's exposure, the same white as on the monitor
        z, c, p = xf if xf is not None else (1.0, (0.0, 0.0), (0.0, 0.0))
        for ch, x, y, size, t0, x0 in items:
            u = np.clip((tau - t0) / 1.05, 0, 1)
            if u <= 0:
                continue
            e = 1 - (1 - u) ** 3
            py = 1.28 + (y - 1.28) * e
            px = x0 + (x - x0) * e
            px, py = (px - c[0]) * z + p[0], (py - c[1]) * z + p[1]
            size = size * z
            k = float(smooth(.62, .98, u))
            area = float(self._area(size))
            if k < 1:
                out.append([px, py, size, cell_of("雨"), 0.0, .62 * white * area, .74 * white * area, white * area, 1 - k, cell_of("雨")])
            if k > 0:
                out.append([px, py, size, cell_of(ch), 0.0, white * area, white * area, white * area * 1.03, k, cell_of(ch)])
        return np.array(out, "f4") if out else None
