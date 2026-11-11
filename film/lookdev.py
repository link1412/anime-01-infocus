"""Look-develop one world through the whole light path (camera body, EVF OLED, eyepiece, phone).

  python -m film.lookdev sea still 18.0 20.5 --out out/sea_a --debug
  python -m film.lookdev sea video 18 22 --out out/sea.mp4

A world module film/worlds/<name>.py exposes class <Name>(gl, size) with
  render(st: dict, out: Target)     writes the frame into `out` (RGBA16F, 1696x954)
  lookdev(t) -> dict                {"world": st, "isp": {...}, "hud": {...}, "ph": {...}}
and optionally draw_hud(hud_state) -> RGBA uint8 array (768, 1024, 4)
"""
from __future__ import annotations
import argparse, importlib, pathlib, subprocess, time
import numpy as np
from PIL import Image
from .core import GL
from .hud import Hud
from .optics import Optics
from .hand import Hands
from .timeline import WORLD, OVER

FPS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("mode", choices=["still", "video"])
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--out", default="out/ld")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--sim", type=int, default=2)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    size = tuple(int(x) for x in a.size.split("x"))
    gl = GL()
    mod = importlib.import_module(f"film.worlds.{a.world}")
    w = getattr(mod, a.world.capitalize())(gl, WORLD)
    target = gl.target(WORLD)
    optics = Optics(gl, out=size, sim=a.sim)
    hud = Hud()
    hands = Hands(120.0)
    draw = getattr(w, "draw_hud", hud.draw)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def step(t, f):
        s = w.lookdev(t)
        w.render(s["world"], target)
        h = {"t": t, **s.get("hud", {})}
        isp = {"roll_small": 0.0, **s.get("isp", {})}
        hh = hands.at(t)
        isp.setdefault("ibis", (hh["yaw"], hh["pitch"]))
        optics.body(target, draw(h), isp, f, OVER)
        optics.phone(s.get("ph", {}), hands, t, f)

    if a.mode == "still":
        for t in a.times:
            optics.fresh, optics.gain = True, None
            t0 = time.time()
            f0 = int(t * FPS)
            for i in range(a.warm, -1, -1):
                step((f0 - i) / FPS, f0 - i)
            optics.final.fbo.read(components=3)
            name = f"{out}_{t:06.2f}.png"
            Image.fromarray(optics.read()).save(name)
            print(name, f"{(time.time() - t0) / (a.warm + 1):.3f}s/frame gain={optics.gain:.2f}")
            if a.debug:
                d = np.frombuffer(target.fbo.read(components=3, dtype="f4"), "f4").reshape(WORLD[1], WORLD[0], 3)
                Image.fromarray((np.flipud(np.clip(d, 0, 1)) ** (1 / 2.2) * 255).astype("u1")).save(f"{out}_{t:06.2f}_world.png")
                p = np.frombuffer(optics.panel.fbo.read(components=3), "u1").reshape(768, 1024, 3)
                Image.fromarray(np.flipud(p)).save(f"{out}_{t:06.2f}_panel.png")
    else:
        t0, t1 = a.times[0], a.times[1]
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                               "-r", str(FPS), "-i", "-", "-vf", "vflip,format=yuv420p", "-c:v", "libx264", "-crf", "16", str(out)], stdin=subprocess.PIPE)
        for f in range(int(t0 * FPS) - 6, int(t1 * FPS)):
            step(f / FPS, f)
            if f >= int(t0 * FPS):
                ff.stdin.write(optics.final.fbo.read(components=3, dtype="f1", alignment=1))
        ff.stdin.close()
        ff.wait()
        print("done", out)


if __name__ == "__main__":
    main()
