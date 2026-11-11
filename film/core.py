"""Thin moderngl helpers: targets, programs, uniform binding."""
from __future__ import annotations
import pathlib, re
import moderngl
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SHADERS = HERE / "shaders"

FS_VERT = """#version 410
in vec2 p; out vec2 uv;
void main(){ uv = p*.5+.5; gl_Position = vec4(p,0.,1.); }
"""


def glsl(name: str) -> str:
    src = (SHADERS / name).read_text()
    return re.sub(r'^#include\s+"(.+?)"\s*$', lambda m: glsl(m.group(1)), src, flags=re.M)


class Target:
    def __init__(self, gl, size, comps=4, dtype="f2", mips=False, nearest=False, clamp=True):
        self.size = tuple(size)
        self.tex = gl.ctx.texture(self.size, comps, dtype=dtype)
        self.tex.repeat_x = self.tex.repeat_y = not clamp
        self.mips = mips
        if nearest:
            self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        elif mips:
            self.tex.build_mipmaps()
            self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = gl.ctx.framebuffer([self.tex])

    def gen(self):
        if self.mips:
            self.tex.build_mipmaps()

    def clear(self, *c):
        self.fbo.clear(*(c or (0, 0, 0, 0)))


class GL:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=410)
        self.tri = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
        self.quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._progs: dict = {}
        self._vaos: dict = {}

    def prog(self, frag: str, vert: str | None = None):
        key = (frag, vert)
        if key not in self._progs:
            vs = glsl(vert) if vert else FS_VERT
            self._progs[key] = self.ctx.program(vertex_shader=vs, fragment_shader=glsl(frag))
        return self._progs[key]

    def target(self, size, **kw) -> Target:
        return Target(self, size, **kw)

    def bind(self, prog, u: dict):
        unit = 0
        for k, v in u.items():
            m = prog.get(k, None)
            if m is None:
                continue
            if isinstance(v, Target):
                v = v.tex
            if isinstance(v, moderngl.Texture):
                v.use(unit)
                m.value = unit
                unit += 1
            else:
                m.value = v

    def run(self, frag: str, target: Target, **u):
        prog = self.prog(frag)
        target.fbo.use()
        self.bind(prog, u)
        key = id(prog)
        if key not in self._vaos:
            self._vaos[key] = self.ctx.simple_vertex_array(prog, self.tri, "p")
        self._vaos[key].render(moderngl.TRIANGLES, vertices=3)

    def instanced(self, prog, fmt: str, names: list[str], vbo):
        return self.ctx.vertex_array(
            prog, [(self.quad, "2f", "p"), (vbo, fmt, *names)], skip_errors=True
        )


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Track:
    """Piecewise keyframes: [(t, value), ...] with smoothstep easing between keys."""

    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k[0])

    def __call__(self, t):
        k = self.keys
        if t <= k[0][0]:
            return k[0][1]
        for (a, va), (b, vb) in zip(k, k[1:]):
            if t <= b:
                return va + (vb - va) * smooth(a, b, t)
        return k[-1][1]
