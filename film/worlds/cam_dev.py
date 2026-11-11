"""lookdev for the camera in the glass: python -m film.worlds.cam_dev out/cam 256 [ev]"""
import sys, time
import numpy as np
from PIL import Image
from ..core import GL
from .portrait import camera_plate

WORLD = (1696, 954)


def tone(x):
    x = np.maximum(x, 0)
    y = (x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def main():
    out = sys.argv[1]
    spp = int(sys.argv[2])
    ev = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    gl = GL()
    cam = dict(uO=(.012, -.006, 1.0), uF=(0., 0., -1.), uR=(1., 0., 0.), uU=(0., 1., 0.), uTan=.205, uAspect=WORLD[0] / WORLD[1])
    t0 = time.time()
    a = camera_plate(gl, WORLD, cam, spp=spp, cache=False)
    print("render", round(time.time() - t0, 1), flush=True)
    im = np.flipud(a)
    rgb = tone(im[..., :3] * im[..., 3:] * 2 ** ev)
    Image.fromarray((rgb * 255).astype("u1")).save(out + ".png")
    H, W = rgb.shape[:2]
    x0, y0 = W // 2 - 300, H // 2 - 170
    crop = Image.fromarray((rgb[y0:y0 + 340, x0:x0 + 600] * 255).astype("u1"))
    crop.resize((1800, 1020), Image.LANCZOS).save(out + "_crop.png")


if __name__ == "__main__":
    main()
