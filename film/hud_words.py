"""The display in the world of text: there are no icons here, only their names."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from . import fonts

S = 2
W, H = 1024, 768
TOP, BOT = 96, 672
WHITE = (236, 232, 222, 255)
DIM = (170, 166, 158, 255)
INK = (46, 41, 34)
PAPER = (221, 207, 180)


class HudWords:
    def __init__(self):
        self.big = fonts.font("Songti.ttc", 27 * S, 6)
        self.med = fonts.font("Songti.ttc", 22 * S, 6)
        self.small = fonts.font("Songti.ttc", 18 * S, 6)
        self.tiny = fonts.font("Songti.ttc", 11 * S, 6)

    def _t(self, d, x, y, s, f, c=WHITE, anchor="ls"):
        # while the panel is still paper, the same words are ink
        k = self.paper
        if k > 0:
            c = tuple(int(a + (b - a) * k) for a, b in zip(c[:3], INK)) + (c[3],)
        d.text((x * S, y * S), s, font=f, fill=c, anchor=anchor)

    def draw(self, s: dict) -> np.ndarray:
        img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        self.paper = float(s.get("paper", 0.0))
        if self.paper > 0:
            fill = PAPER + (int(255 * self.paper),)
            d.rectangle([0, 0, W * S, TOP * S], fill=fill)
            d.rectangle([0, BOT * S, W * S, H * S], fill=fill)
        # area 1: audio, focus mode, standby and time, format, card, battery
        self._t(d, 26, 54, "左声 右声", self.small, DIM)
        self._t(d, 246, 54, "连续对焦", self.small)
        self._t(d, 440, 54, "待机", self.med, anchor="ms")
        self._t(d, 490, 54, "零时零分零秒", self.small)
        self._t(d, 620, 54, "四开", self.small)
        self._t(d, 660, 54, "廿四格", self.small)
        self._t(d, 722, 54, "卡一", self.small)
        self._t(d, 766, 54, "余二时十四分", self.small)
        self._t(d, 998, 54, "电量八十四", self.small, anchor="rs")
        # what shows on standby
        self._t(d, 26, 85, "手动", self.small)
        self._t(d, 74, 85, "广域", self.small)
        self._t(d, 136, 85, "字", self.small, (255, 196, 120, 255))
        self._t(d, 170, 85, "防抖开", self.small)
        # area 2: shutter, aperture, exposure, sensitivity, white balance
        y = 706
        self._t(d, 30, y, "焦", self.big, DIM)
        self._t(d, 86, y, "五十分之一", self.big)
        self._t(d, 250, y, "一点八", self.big)
        self._t(d, 380, y, "减三二一〇一二三加", self.med, DIM)
        self._t(d, 640, y, "零", self.big)
        self._t(d, 700, y, "感光六千四百", self.big)
        self._t(d, 998, y, "自动白平衡", self.small, anchor="rs")
        self._t(d, 86, 735, "零时零分零秒零格", self.small, DIM)
        self._t(d, 250, 735, "五十五毫米", self.small, DIM)
        if s.get("mag"):
            self._t(d, 998, 660, "放大" + s["mag"]["label"], self.big, anchor="rs")
        a = s.get("af")
        if a:
            px, py = a["x"] * W, TOP + a["y"] * (BOT - TOP)
            w, h = a["w"], a["h"]
            al = int(255 * a.get("alpha", 1.0))
            col = (255, 90, 70, al) if a.get("state") == "fail" else (245, 240, 230, al)
            step = 11
            nx, ny = max(int(w / step), 2), max(int(h / step), 2)
            for i in range(nx + 1):
                for yy in (py - h / 2, py + h / 2):
                    self._t(d, px - w / 2 + i * w / nx, yy, "框", self.tiny, col, anchor="mm")
            for i in range(1, ny):
                for xx in (px - w / 2, px + w / 2):
                    self._t(d, xx, py - h / 2 + i * h / ny, "框", self.tiny, col, anchor="mm")
        return np.asarray(img.resize((W, H), Image.LANCZOS), dtype="u1")
