#!/usr/bin/env python3
"""sheet.py out.png cols w file...: contact sheet for reviewing frames."""
import sys
from PIL import Image
out, cols, w, *names = sys.argv[1:]
cols, w = int(cols), int(w)
ims = [Image.open(n).convert("RGB") for n in names]
h = int(w * ims[0].height / ims[0].width)
rows = (len(ims) + cols - 1) // cols
S = Image.new("RGB", (cols * w, rows * h))
for i, m in enumerate(ims):
    S.paste(m.resize((w, h), Image.LANCZOS), ((i % cols) * w, (i // cols) * h))
S.save(out)
