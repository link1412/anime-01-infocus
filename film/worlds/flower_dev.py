"""lookdev: python -m film.worlds.flower_dev out/fl t fd [zoom gcx gcy ev passes] [key=value ...] [stub]"""
import sys, time
import numpy as np
from PIL import Image
from ..core import GL
from .sea import Sea
from .flower import Flower

WORLD = (1696, 954)


def tone(x):
    x = np.maximum(x, 0)
    y = (x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def main():
    a = sys.argv[1:]
    stub = "stub" in a
    a = [x for x in a if x != "stub"]
    extra = {}
    for x in [x for x in a if "=" in x]:
        k, v = x.split("=")
        extra[k] = float(v)
    a = [x for x in a if "=" not in x]
    out = a[0]
    t, fd = float(a[1]), float(a[2])
    zoom = float(a[3]) if len(a) > 3 else 1.0
    gc = (float(a[4]), float(a[5])) if len(a) > 5 else (0.0, 0.0)
    ev = float(a[6]) if len(a) > 6 else 0.0
    passes = int(a[7]) if len(a) > 7 else 64
    gl = GL()
    t0 = time.time()
    sea = None if stub else Sea(gl, WORLD)
    fl = Flower(gl, WORLD, sea=sea)
    print("setup", round(time.time() - t0, 1), flush=True)
    w = gl.target(WORLD)
    for i in range(2):
        t0 = time.time()
        fl.render(dict(t=t, fd=fd, zoom=zoom, gc=gc, passes=passes, end=(0.0, -.095, .27), **extra), w)
        gl.ctx.finish()
        print("frame", round(time.time() - t0, 2), flush=True)
    im = np.frombuffer(w.fbo.read(components=3, dtype="f4"), "f4").reshape(WORLD[1], WORLD[0], 3)
    Image.fromarray((tone(np.flipud(im) * 2 ** ev) * 255).astype("u1")).save(out + ".png")


if __name__ == "__main__":
    main()
