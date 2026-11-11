"""Make the whole film: the sound first (the camera's level meters read it), then the picture, then the two together.

    python -m film.make out/hejiao.mp4

Run from the repository root. Needs ffmpeg on PATH; the audio encoder (aac_at) is macOS's.
"""
from __future__ import annotations
import argparse, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIX = "out/audio/mix.wav"
MASTER = "volume=11dB,alimiter=level_in=1:level_out=1:limit=0.81:attack=4:release=90:asc=0:level=0"


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="out/hejiao.mp4")
    ap.add_argument("--crf", type=int, default=17)
    a = ap.parse_args()
    from .timeline import FILM_END

    out = (pathlib.Path.cwd() / a.out).resolve()
    picture = out.with_name(out.stem + "_picture.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run([sys.executable, "-m", "film.audio", MIX])
    run([sys.executable, "-m", "film.render", "video", "0", f"{FILM_END:.2f}", "--out", str(picture), "--crf", str(a.crf)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(picture), "-i", MIX, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-af", MASTER, "-c:a", "aac_at", "-b:a", "256k", "-movflags", "+faststart", "-shortest", str(out)])
    print(f"done {out}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
