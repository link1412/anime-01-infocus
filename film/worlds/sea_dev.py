"""Development helper (world only, fast): python -m film.worlds.sea_dev out/x 18 --crop x0,y0,w,h"""
from __future__ import annotations
import argparse, time, json
import numpy as np
from PIL import Image
from ..core import GL
from .sea import Sea


def tone(x):
    y = (x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14)
    return np.clip(y, 0, 1)


def to_png(d, ev):
    e = np.maximum(d, 0) * 2 ** ev
    t = tone(e)
    s = np.where(t <= .0031308, 12.92 * t, 1.055 * t ** (1 / 2.4) - .055)
    return (np.flipud(s) * 255 + .5).astype("u1")


def grab(w, tgt, st):
    w.render(st, tgt)
    return np.frombuffer(tgt.fbo.read(components=3, dtype="f4"), "f4").reshape(954, 1696, 3).copy()


def contract(w, tgt):
    base = dict(t=24.5, yaw=.75, pitch=-.095, roll=0.0, tan_half=.27)
    a = grab(w, tgt, base)
    b = grab(w, tgt, dict(base, crack=dict(t_arrive=22.25, edge_az=.42)))
    print("crack after load: max abs diff", np.abs(a - b).max())
    late = dict(base, t=22.25 + 1.4 + .55 + .01)
    a = grab(w, tgt, late)
    b = grab(w, tgt, dict(late, crack=dict(t_arrive=22.25, edge_az=.42)))
    print("crack right after last tile: max abs diff", np.abs(a - b).max())
    # bead -> normal continuity
    t = 16.0
    a = grab(w, tgt, dict(t=t, yaw=0.0, pitch=-.095, roll=np.pi, tan_half=.27))
    b = grab(w, tgt, dict(t=t, yaw=0.0, pitch=-.095, roll=0.0, tan_half=.27, drop=dict(cx=0.0, cy=0.0, R=3.09, lens=0.0)))
    d = np.abs(a - b)
    print("bead vs normal at R=3.09: max diff", d.max(), "mean", d.mean(), "p99.9", np.percentile(d, 99.9))
    Image.fromarray(to_png(np.maximum(a, 0), -1.05)).save("out/sea_c_normal.png")
    Image.fromarray(to_png(np.maximum(b, 0), -1.05)).save("out/sea_c_bead.png")
    # determinism: render same frame twice after other frames
    a = grab(w, tgt, dict(base, t=18.0, yaw=0.0))
    grab(w, tgt, dict(base, t=19.0, yaw=0.3))
    b = grab(w, tgt, dict(base, t=18.0, yaw=0.0))
    print("determinism: max abs diff", np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--ev", type=float, default=-.55)
    ap.add_argument("--set", action="append", default=[], help="key=value overrides of the world state")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--contract", action="store_true")
    a = ap.parse_args()
    gl = GL()
    t0 = time.time()
    w = Sea(gl, (1696, 954))
    print(f"init {time.time() - t0:.2f}s", flush=True)
    print(w.ocean.describe())
    print("sun_e", w.sun_e, "sun_l", w.sun_l)
    tgt = gl.target((1696, 954))
    if a.contract:
        contract(w, tgt)
        return
    for t in a.times:
        s = w.lookdev(t)["world"]
        s.update({kv.split("=")[0]: json.loads(kv.split("=")[1]) for kv in a.set})
        t1 = time.time()
        w.render(s, tgt)
        d = np.frombuffer(tgt.fbo.read(components=3, dtype="f4"), "f4").reshape(954, 1696, 3)
        el = time.time() - t1
        name = f"{a.out}_{t:06.2f}.png"
        Image.fromarray(to_png(d, a.ev)).save(name)
        print(name, f"{el * 1000:.0f} ms")
        if a.stats:
            rows = [("sky top", 880, 200), ("sky mid", 600, 300), ("hor+", 510, 200), ("hor-", 470, 200), ("sea mid", 300, 200), ("sea low", 60, 200),
                    ("glit hi", 470, 848), ("glit lo", 150, 848)]
            for name_, y, x in rows:
                print(f"  {name_:8s}", d[y - 4:y + 4, x - 4:x + 4].mean((0, 1)).round(3))
            print("  max", d.max((0, 1)).round(1))


if __name__ == "__main__":
    main()
