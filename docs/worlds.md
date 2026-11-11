# Writing a world

A world is whatever the camera is pointed at. It renders one linear-light frame. Everything after that frame belongs to the shared light path, and a world never touches it:

```
world (1696x954 RGBA16F)
  -> film/shaders/isp.frag        camera body: exposure, picture profile, HUD composite
  -> OLED panel 1024x768          the EVF, with its subpixels
  -> eyepiece                     optics in front of the panel
  -> phone sensor                 film/shaders/phone_sensor.frag (Bayer mosaic, noise)
  -> demosaic / TNR / finish      the phone's own processing
```

## Contract

`film/worlds/<name>.py` defines `class <Name>(gl, size)`. The class name is the module name with its first letter capitalised. It has:

| method | what it does |
|---|---|
| `render(st: dict, out: Target)` | writes the frame into `out` (RGBA16F, 1696x954) |
| `lookdev(t) -> dict` | returns `{"world": st, "isp": {...}, "hud": {...}, "ph": {...}}` for a standalone look at the world |
| `draw_hud(hud_state)` (optional) | returns an RGBA uint8 array (768, 1024, 4) when the world needs its own viewfinder overlay |

`desk.py` is a compact example. `sea.py`, `ink.py` and `pixel.py` are the larger ones.

Look at a world on its own, through the full light path:

```sh
python -m film.lookdev sea still 18.0 20.5 --out out/sea --debug   # --debug also writes the world and panel images
python -m film.lookdev sea video 18 22 --out out/sea.mp4
```

## Coordinates and helpers

- Frame size is 1696x954. In "view units", y runs over [-1, 1] and x over [-1.778, 1.778] across the whole target. The camera crops the central 1/1.06 of that and keeps the rest as overscan for hand shake.
- `film/core.py` has `GL`, `Target`, `Track` (keyframes with smoothstep easing) and `smooth`. Shared GLSL lives in `film/shaders/common.glsl`.
- Shaders are `#version 410`. `#include "file.glsl"` is resolved by `core.glsl()`. For instanced quads, use `gl.instanced`.
- Text goes through `film/fonts.py`, which finds the macOS system faces (`Songti.ttc`, `Kaiti.ttc`, `Xingkai.ttc`, `PingFang.ttc`, `Menlo.ttc`, `Helvetica.ttc`, `HelveticaNeue.ttc`, `Arial Narrow.ttf`). The pixel world's HUD type is drawn from bitmap tables in `hud_pixel.py`. Fonts are the only files the film reads that it did not make itself.

## Rules

- **Deterministic and seekable.** A frame depends only on `st` and never on the frames before it. If something has to be simulated, precompute it and cache it under `out/cache/`, keyed by a hash of whatever produced it (see `beach_sim._key`, `sea_sky._key`, `ink_scroll.source_key`). `flower_geo.py` is the one exception: it is versioned by hand through `VERSION`, so bump that if you change the geometry.
- **Budget.** Aim for about 0.3 s a frame on an M-series GPU. The finished film averages about 0.17 s.
- **Judge the result through the phone.** Look at the final phone frame and at the world image, both at full size. To check motion, render nearby frames and compare them, or build a contact sheet with `python sheet.py out.png <cols> <width> frame*.png`.
- No logos or brand names in the image, and nothing that recreates an existing artwork, film or game.
