"""A window at night, seen from far away in a dark pane of glass. Computed, like everything else here:
an overcast town's light on low cloud, glazing bars, a curtain drawn part way, rain running down the outside."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from ..core import GL, Target

# seen in the monitor by a camera a metre away, the window's mirror image is three metres off
METRES_PER_UNIT = 3.0 * .205


def _fbm(shape, rng, base, oct_=5, aspect=1.0):
    h, w = shape
    out = np.zeros(shape, "f4")
    a, tot = 1.0, 0.0
    for o in range(oct_):
        nx, ny = int(base * 2 ** o * aspect) + 2, int(base * 2 ** o) + 2
        g = rng.random((ny, nx), dtype="f4")
        out += a * ndimage.zoom(g, (h / ny, w / nx), order=3)[:h, :w]
        tot += a
        a *= .5
    return out / tot


def plate(size, seed=23):
    """linear rgb, image coordinates y up."""
    W, H = size
    aspect = W / H
    rng = np.random.default_rng(seed)
    x = (np.arange(W) + .5) / W * 2 * aspect - aspect
    y = (np.arange(H) + .5) / H * 2 - 1
    X, Y = np.meshgrid(x.astype("f4"), y.astype("f4"))
    # in room metres, as the glass shows them
    mx, my = X * METRES_PER_UNIT + .012, Y * METRES_PER_UNIT - .006
    x0, y0, x1, y1 = -.72, -.12, .42, 1.32
    clouds = _fbm((H, W), rng, 3, 5, aspect)
    t = np.clip((my - y0) / (y1 - y0), 0, 1)[..., None]
    # a town's light on the underside of low cloud: dull amber low down, slate above it
    sky = np.array((.26, .17, .105), "f4") * (1 - t) ** 1.8 + np.array((.070, .092, .135), "f4") * (.45 + .55 * t)
    sky = sky * (.62 + .76 * clouds[..., None])
    # two lamps a long way off, nothing more than places where the rain is brighter
    for lx, ly, s, c in ((-.33, .06, .13, (.34, .205, .10)), (.09, -.05, .07, (.22, .19, .15))):
        sky = sky + np.exp(-((mx - lx) ** 2 + (my - ly) ** 2 / 1.3) / (2 * s * s))[..., None] * np.array(c, "f4")
    # water on the outside of the pane: wandering runs and fat beads, each a little lens on the glow behind it
    run = np.zeros((H, W), "f4")
    for _ in range(90):
        px, py = rng.uniform(0, W), rng.uniform(0, H)
        a = rng.uniform(-1., 1.)
        wander = rng.uniform(.15, .8)
        for j in range(int(rng.uniform(90, 520))):
            py -= 1
            if py < 0:
                break
            px += rng.normal(0, wander) * .6
            run[int(py), int(px) % W] += a
    run = ndimage.gaussian_filter(run, (3., 4.5)) * 16
    beads = np.zeros((H, W), "f4")
    n = 620
    beads[rng.integers(0, H, n), rng.integers(0, W, n)] = rng.uniform(-1., 1., n)
    beads = ndimage.gaussian_filter(beads, 5.5) * 70
    sky = sky * np.clip(1 + run + beads, .35, 1.9)[..., None]
    # the frame and its glazing bars, then the curtain drawn across the right hand side
    def box(u0, u1, v0, v1, e=.012):
        return np.clip((mx - u0) / e, 0, 1) * np.clip((u1 - mx) / e, 0, 1) * np.clip((my - v0) / e, 0, 1) * np.clip((v1 - my) / e, 0, 1)
    bar_x = np.clip(np.abs(mx - .5 * (x0 + x1)) / .018 - .6, 0, 1)
    bar_y = np.clip(np.abs(my - .42) / .016 - .6, 0, 1)
    edge = .17 + .035 * np.sin(my * 3.3 + .8)
    folds = .55 + .45 * np.sin((mx - edge) * 62 + 1.4 * np.sin(my * 2.3))
    curtain = np.clip((mx - edge) / .03, 0, 1)
    mask = box(x0, x1, y0, y1) * bar_x * bar_y * (1 - curtain * .94) * (1 - curtain * .35 * folds)
    img = sky * mask[..., None]
    # the rest of the wall. two weak lights reach it: the monitor, straight on from across the room, cold and strongest
    # in the middle, and what comes in through the window, warm and only close by
    cold = np.exp(-(mx ** 2 / 1.5 + (my + .05) ** 2 / 1.0))[..., None] * np.array((.46, .55, .74), "f4")
    near = np.exp(-np.maximum(np.abs(mx + .15) - .57, 0) ** 2 / .10 - np.maximum(np.abs(my - .6) - .72, 0) ** 2 / .07)
    warm = near[..., None] * np.array((.40, .31, .24), "f4")
    light = cold + warm
    plaster = (.86 + .28 * _fbm((H, W), rng, 9, 4, aspect))[..., None]
    room = light * .085 * plaster
    # the sill catches what falls on it
    sill = box(x0 - .06, x1 + .06, y0 - .045, y0 + .004, .006)
    shadow = np.clip(1 - np.exp(-np.maximum(y0 - .045 - my, 0) / .05), .45, 1) * (1 - sill)
    room = room * (shadow[..., None] * box(x0 - .06, x1 + .06, -9, y0, .02)[..., None] + (1 - box(x0 - .06, x1 + .06, -9, y0, .02)[..., None]))
    room += sill[..., None] * np.array((.17, .125, .092), "f4") * (.8 + .4 * clouds[..., None])
    # under it a radiator, its fins
    rad = box(x0 + .02, x1 - .10, -.86, -.30, .01)
    fins = .55 + .45 * np.cos((mx - x0) * 2 * np.pi / .052)
    room = room * (1 - rad[..., None]) + rad[..., None] * light * (.045 + .075 * fins[..., None] * (.75 + .25 * np.clip((my + .86) / .56, 0, 1))[..., None])
    # the curtain, gathered where it was pulled to: folds that catch the screen one side and the window the other
    c_end = edge + .58
    cm = np.clip((mx - edge) / .025, 0, 1) * np.clip((c_end - mx) / .05, 0, 1)
    gather = 1 + 1.5 * np.clip((mx - edge) / .58, 0, 1)
    f = (mx - edge) * 40 * gather + 1.4 * np.sin(my * 2.3) + .9 * np.sin(my * 5.1 + mx * 9.)
    f1 = .5 + .5 * np.sin(f)
    f2 = .5 + .5 * np.sin(f * 2.3 + 1.1)
    drape = .22 + .62 * f1 ** 1.6 + .16 * f2
    glow = np.exp(-np.maximum(mx - edge, 0) / .16)[..., None] * np.array((.30, .19, .12), "f4") * (1 - .2 * np.clip(-(my + .1), 0, 1))[..., None] * np.clip((my + .95) / .6, 0, 1)[..., None]
    room = room * (1 - cm[..., None]) + cm[..., None] * (cold * .10 + glow * .55) * drape[..., None]
    # a bookcase against the wall on the far side of it
    bx0, bx1 = .85, 1.62
    case = box(bx0, bx1, -1.2, .98, .008)
    books = np.zeros_like(img)
    def rect(u0, u1, v0, v1, c):
        i0, i1 = [int(np.clip(((u - .012) / METRES_PER_UNIT + aspect) / (2 * aspect) * W, 0, W)) for u in (u0, u1)]
        j0, j1 = [int(np.clip(((v + .006) / METRES_PER_UNIT + 1) / 2 * H, 0, H)) for v in (v0, v1)]
        books[j0:j1, i0:i1] = c
    SPINES = np.array([(.34, .10, .08), (.09, .14, .24), (.42, .36, .24), (.12, .20, .15), (.46, .44, .40), (.20, .17, .16), (.40, .24, .10), (.06, .06, .07)], "f4")
    rows = np.arange(-1.14, .98, .315)
    for v0 in rows:
        rect(bx0, bx1, v0 - .024, v0, np.array((.20, .15, .10), "f4"))       # the board
        u = bx0 + .018
        while u < bx1 - .02:
            w = float(rng.uniform(.017, .046))
            if rng.random() < .10:
                u += float(rng.uniform(.03, .11))                            # a gap, or something lying flat
                continue
            ht = float(rng.uniform(.17, .27))
            rect(u, u + w - .0015, v0, v0 + ht, SPINES[rng.integers(len(SPINES))] * float(rng.uniform(.55, 1.15)))
            u += w
    room = room * (1 - case[..., None]) + case[..., None] * (books + .012) * light * .26
    img = img + room * (1 - mask[..., None])
    return ndimage.gaussian_filter(img, (2., 2., 0)).astype("f4")


def window_target(gl: GL, size) -> Target:
    t = gl.target(size, mips=True)
    a = plate(size)
    rgba = np.concatenate([a, np.ones(a.shape[:2] + (1,), "f4")], -1)
    t.tex.write(rgba.astype("f2").tobytes())
    t.gen()
    return t
