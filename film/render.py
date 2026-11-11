"""Render stills or video through the whole light path.
  python -m film.render still 3.0 6.5 --out out/x --debug
  python -m film.render video 0 16 --out out/seg.mp4 [--size 1920x1080] [--sim 2]
"""
from __future__ import annotations
import argparse, pathlib, subprocess, time
import numpy as np
from PIL import Image, ImageDraw
from .core import GL
from . import fonts
from .hud import Hud
from .optics import Optics
from .timeline import Film, OVER, T_END, TITLE_AT, TITLE_OUT, CARD_AT, CARD_OUT

FPS = 30


def title_card(size, t):
    """written straight to the file: nothing filmed this."""
    img = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    face = lambda k: fonts.font("Songti.ttc", int(size[1] * k), 6)
    cx, cy = size[0] / 2, size[1] / 2
    if TITLE_AT <= t < TITLE_OUT:
        draw.text((cx, cy), "《合焦》", font=face(.040), fill=(214, 214, 214), anchor="mm")
    elif CARD_AT <= t < CARD_OUT:
        draw.text((cx, cy - size[1] * .036), "本片没有摄影机。", font=face(.036), fill=(214, 214, 214), anchor="mm")
        draw.text((cx, cy + size[1] * .036), "没有一张照片，没有一段录音，全部由 Claude 写的代码算出。", font=face(.030), fill=(150, 150, 150), anchor="mm")
    return img


def pipeline(size, sim):
    gl = GL()
    film = Film(gl)
    optics = Optics(gl, out=size, sim=sim)
    hud = Hud()
    return gl, film, optics, hud


def step(film, optics, hud, t, frame):
    s = film.state(t)
    world = film.render_world(s, optics, frame)
    rgba = film.draw_hud(s, hud)
    optics.body(world, rgba, s["isp"], frame, s["isp"].get("over", OVER))
    optics.phone(s["ph"], film.hands, t, frame)
    film.after_body(s, optics, frame)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["still", "video"])
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--out", default="out/still")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--sim", type=int, default=2)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--crf", type=int, default=15)
    a = ap.parse_args()
    size = tuple(int(x) for x in a.size.split("x"))
    gl, film, optics, hud = pipeline(size, a.sim)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "still":
        for t in a.times:
            if t >= T_END:
                title_card(size, t).save(f"{out}_{t:06.2f}.png")
                continue
            optics.fresh = True
            optics.gain = None
            t0 = time.time()
            f0 = int(t * FPS)
            for i in range(a.warm, -1, -1):
                f = f0 - i
                step(film, optics, hud, f / FPS, f)
            name = f"{out}_{t:06.2f}.png"
            Image.fromarray(optics.read()).save(name)
            print(name, f"{(time.time() - t0) / (a.warm + 1):.3f}s/frame gain={optics.gain:.2f}")
            if a.debug:
                w = np.frombuffer(film.world.fbo.read(components=3, dtype="f4"), "f4").reshape(film.world.size[1], film.world.size[0], 3)
                Image.fromarray((np.flipud(np.clip(w, 0, 1)) ** (1 / 2.2) * 255).astype("u1")).save(f"{out}_{t:06.2f}_world.png")
                p = np.frombuffer(optics.panel.fbo.read(components=3), "u1").reshape(768, 1024, 3)
                Image.fromarray(np.flipud(p)).save(f"{out}_{t:06.2f}_panel.png")
    else:
        t0, t1 = a.times[0], a.times[1]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(FPS), "-i", "-", "-vf", "vflip,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-crf", str(a.crf), "-movflags", "+faststart", str(out)]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        start = time.time()
        f0, f1 = int(t0 * FPS), int(t1 * FPS)
        for f in range(f0 - 8, f1):
            if f / FPS >= T_END:
                if f >= f0:
                    ff.stdin.write(np.flipud(np.asarray(title_card(size, f / FPS))).tobytes())
                continue
            step(film, optics, hud, f / FPS, f)
            if f >= f0:
                data = optics.final.fbo.read(components=3, dtype="f1", alignment=1)
                ff.stdin.write(data)
            if f % 30 == 0:
                el = time.time() - start
                print(f"t={f / FPS:.1f}s  {el / max(f - f0 + 9, 1):.3f}s/frame gain={optics.gain:.2f}", flush=True)
        ff.stdin.close()
        ff.wait()
        print("done", out, f"{time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
