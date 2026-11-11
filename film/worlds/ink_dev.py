"""Dev: render world states straight (display-referred, no optics), with EVF-resolution copies.
  python -m film.worlds.ink_dev out/inkdev a:pan=0 b:pan=1.2,blur=.4 --crop .. --rebake near,fore
"""
from __future__ import annotations
import argparse, time
import numpy as np
from PIL import Image
from ..core import GL
from .ink import Ink

WORLD = (1696, 954)
OVER = 1.06


def to_rgb(target):
    d = np.frombuffer(target.fbo.read(components=3, dtype="f4"), "f4").reshape(WORLD[1], WORLD[0], 3)
    return (np.flipud(np.clip(d, 0, 1)) * 255 + .5).astype("u1")


def to_panel(rgb):
    h, w = rgb.shape[:2]
    cw, ch = w / OVER, h / OVER
    x0, y0 = (w - cw) / 2, (h - ch) / 2
    return np.asarray(Image.fromarray(rgb).resize((1024, 576), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("states", nargs="+")
    ap.add_argument("--rebake", default="")
    ap.add_argument("--keep", action="store_true", help="reuse caches regardless of source hash, except --rebake")
    ap.add_argument("--crop", default="")
    ap.add_argument("--panel", action="store_true")
    a = ap.parse_args()
    gl = GL()
    t0 = time.time()
    rb = tuple(x for x in a.rebake.split(",") if x)
    w = Ink(gl, WORLD, rebake=rb)
    print(f"init {time.time() - t0:.1f}s", flush=True)
    target = gl.target(WORLD)
    for s in a.states:
        name, _, kv = s.partition(":")
        st = dict(t=30.0, pan=0.0, pan_speed=0.0, blur=0.0, zoom=1.0, zc=(0.0, 0.0))
        for item in filter(None, kv.split(",")):
            k, _, v = item.partition("=")
            if k == "zc":
                x, y = v.split("/")
                st[k] = (float(x), float(y))
            else:
                st[k] = float(v)
        t1 = time.time()
        if "reveal" in st:
            st["from"] = w.dummy_sea(st["t"])
        w.render(st, target)
        target.fbo.read(components=1, viewport=(0, 0, 1, 1))
        dt = time.time() - t1
        rgb = to_rgb(target)
        Image.fromarray(rgb).save(f"{a.prefix}_{name}.png")
        if a.panel:
            Image.fromarray(to_panel(rgb)).save(f"{a.prefix}_{name}_p.png")
        if a.crop:
            x0, y0, x1, y1 = (int(q) for q in a.crop.split(","))
            Image.fromarray(rgb[y0:y1, x0:x1]).save(f"{a.prefix}_{name}_c.png")
        print(name, f"{dt * 1000:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
