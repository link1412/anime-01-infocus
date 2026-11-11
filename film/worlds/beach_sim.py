"""The water of the beach: a real shallow-water solve on the GPU.

The camera never moves, only turns, so the grid is log-polar around it: a column is an azimuth,
a row is ln(range). Cells are square, and each one is about the same size on screen wherever it
is, so the swash at 20 m gets 6 cm cells while the outer bar at 250 m gets 80 cm cells. Waves cross
that grid on a finite-volume scheme with world-frame momentum (well-balanced by hydrostatic
reconstruction, second order, wet/dry), driven by an irregular wave train at the offshore arc.

State is integrated at a fixed rate from a fixed start, so the state at time t is one thing. To
keep a frame independent of whatever was rendered before it, the integrator keeps checkpoints in
the cache and seeks through them.
"""
from __future__ import annotations
import hashlib, pathlib, time
import numpy as np
import moderngl
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from ..core import GL, glsl

CACHE = pathlib.Path(__file__).resolve().parents[2] / "out" / "cache"

# ---- grid
TH0, TH1 = -.66, 1.50         # azimuth range, rad, 0 = towards the sun, + = right
R0, R1 = 6.0, 520.0          # range, m
D = .0022                     # cell size in theta and in ln(r)
NT = int(round((TH1 - TH0) / D))
NR = int(round(np.log(R1 / R0) / D))
RHO0 = float(np.log(R0))

CAM_H = 4.4                   # camera above mean water

RATE = 360                    # solver steps per second
SCALAR_EVERY = 6              # the foam and wetness update runs at 60 Hz
SPIN = 180.0                  # seconds of waves before t=0
SET_AT = 60.0                 # the swell's clock against the film's: which of its sets comes up the beach during the shot
CKPT = 4.0                    # seconds between checkpoints
WRAP = 120.0                  # film time after which the water repeats
MAT_PERIOD = 2.4              # seconds between restarts of each set of material coordinates


def shore(x):
    """Where the mean waterline is, z(x): a long shallow bay, the far end curling out to the right."""
    return 25.0 + .085 * x + np.where(x > 0, .00105 * x * x, .00028 * x * x)


# across-shore profile: metres offshore of the waterline -> elevation above mean water
_PROFILE = PchipInterpolator(
    [-90, -60, -42, -30, -24, -18, -12, -8, 0, 6, 14, 24, 34, 52, 66, 90, 125, 165, 215, 270, 350, 480, 700],
    [4.4, 4.0, 3.5, 2.75, 2.25, 1.65, 1.0, .56, 0.0, -.4, -.85, -1.15, -1.3, -.98, -1.7, -3.2, -2.6, -4.0, -7.0, -10.0, -13.0, -15.0, -18.0])


def _spectral(n, cell, corr, seed, beta=2.2):
    """Periodic Gaussian noise on an n*n grid of `cell` metres, correlation length about `corr`."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, cell)
    kx, kz = np.meshgrid(f, f, indexing="xy")
    k = np.hypot(kx, kz)
    k0 = 1 / corr
    amp = (1 + (k / k0) ** 2) ** (-beta / 2)
    amp[0, 0] = 0
    w = np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * amp).real
    return (w / w.std()).astype("f4")


class Noise2:
    """Bicubic lookups into a periodic noise tile, in metres."""

    def __init__(self, n, cell, corr, seed, beta=2.2):
        self.a, self.cell, self.n = _spectral(n, cell, corr, seed, beta), cell, n

    def __call__(self, x, z):
        return map_coordinates(self.a, [np.asarray(z) / self.cell, np.asarray(x) / self.cell], order=1, mode="grid-wrap")


def bed_large(x, z):
    """Elevation of the sea floor and the beach, metres above mean water; the scales the water feels."""
    n_bar = Noise2(1024, 1.0, 70.0, 11)
    n_dep = Noise2(1024, 1.0, 38.0, 12)
    n_cusp = Noise2(1024, .5, 9.0, 13, beta=3.0)
    n_rip = Noise2(1024, 1.0, 26.0, 14, beta=3.0)
    n_lump = Noise2(1024, .12, 1.7, 15, beta=2.7)
    zs = shore(x)
    slope = np.gradient(zs, axis=-1) * 0 + (.085 + 2 * np.where(x > 0, .00105, .00028) * x)
    d = (z - zs) / np.sqrt(1 + slope * slope)
    # the bars wander along the beach and the waterline does not quite follow the bay
    wander = 24.0 * n_bar(x * .55, z * .25) * np.clip((d - 12) / 40, 0, 1) + .85 * n_cusp(x * .55, np.zeros_like(x))
    b = _PROFILE(np.clip(d + wander, -90, 700))
    # rip channels cut the bars, the whole bed breathes
    gate = np.clip((d - 18) / 30, 0, 1) * np.clip((300 - d) / 80, 0, 1)
    b += gate * (.42 * n_dep(x, z) - .48 * np.clip(n_rip(x * .45, z * .12) - .7, 0, 2))
    # beach cusps: horns and bays in the swash zone
    cz = np.clip(1 - np.abs(d + 8) / 15, 0, 1)
    b += .14 * cz * n_cusp(x, z * .35)
    # the swash finds every hollow: its edge is lobed because the sand is not a plane
    b += .028 * np.clip((d + 34) / 8, 0, 1) * np.clip((8 - d) / 6, 0, 1) * n_lump(x, z)
    return b


def rocks(x, z):
    """Basalt the water has to go around: the low skerry the figure stands on, reefs, awash boulders."""
    h = np.full_like(x, -99.0)
    n = Noise2(1024, .25, 2.2, 21, beta=2.6)
    for rx, rz, ra, rb, ang, top in ROCKS:
        ca, sa = np.cos(ang), np.sin(ang)
        px, pz = (x - rx) * ca - (z - rz) * sa, (x - rx) * sa + (z - rz) * ca
        q = np.sqrt((px / ra) ** 2 + (pz / rb) ** 2) + .16 * n(x * 1.3 + rx, z * 1.3 + rz)
        a = top * (1 - np.clip(q, 0, 2.2) ** 2.4) + .12 * top * n(x * .8, z * .8)
        h = np.maximum(h, a - 3.0 * np.clip(q - 1, 0, 3))
    return h


# x, z, half-length, half-width, angle, height above mean water
SKERRY = (39.6, 307.0, 13.0, 6.5, .35, 1.9)
ROCKS = [SKERRY,
         (47.0, 322.0, 9.0, 4.0, -.2, .8),
         (28.0, 298.0, 5.0, 3.0, .7, .45)]


def bed_at(x, z):
    return np.maximum(bed_large(x, z), rocks(x, z))


def polar_grid():
    th = TH0 + (np.arange(NT) + .5) * D
    rho = RHO0 + (np.arange(NR) + .5) * D
    T, P = np.meshgrid(th, rho)
    r = np.exp(P)
    return r * np.sin(T), r * np.cos(T), T, r


class Waves:
    """An irregular, directionally spread wave train arriving at the offshore arc."""

    def __init__(self, seed=7, n=44, Hs=3.4, Tp=10.0, heading=.11, spread=.075, depth=15.0):
        rng = np.random.default_rng(seed)
        fp = 1 / Tp
        f = fp * np.exp(rng.uniform(-.26, .5, n))
        # a Pierson-Moskowitz-like skirt with a narrow peak, so sets of bigger waves arrive
        s = f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * 2.4 ** np.exp(-(f - fp) ** 2 / (2 * (.08 * fp) ** 2))
        s = s * np.exp(rng.normal(0, .35, n))
        a = np.sqrt(s / s.sum()) * Hs / 4 * np.sqrt(2)
        ang = np.pi + heading + rng.normal(0, spread, n)             # direction of travel, from +z towards the camera
        c = np.sqrt(9.81 * depth)
        w = 2 * np.pi * f
        self.kx, self.kz = w / c * np.sin(ang), w / c * np.cos(ang)
        self.w, self.a, self.phi = w, a, rng.uniform(0, 2 * np.pi, n)
        self.nx, self.nz = np.sin(ang), np.cos(ang)
        self.g_over_c = 9.81 / c

    def at(self, x, z, t):
        """-> elevation, world velocity of the incident field on the arc."""
        ph = np.outer(x, self.kx) + np.outer(z, self.kz) - self.w * t + self.phi
        e = self.a * np.cos(ph)
        eta = e.sum(1)
        u = (e * self.nx).sum(1) * self.g_over_c
        v = (e * self.nz).sum(1) * self.g_over_c
        return eta, u, v


def _key():
    src = "".join(glsl(n) for n in ("beach_swe.frag", "beach_scalar.frag", "beach_mat.frag")) + pathlib.Path(__file__).read_text()
    return hashlib.sha1(src.encode()).hexdigest()[:12]


class BeachSim:
    def __init__(self, gl: GL):
        self.gl = gl
        ctx = gl.ctx
        size = (NT, NR)
        self.size = size
        self.key = _key()
        self.dir = CACHE / f"beach_sim_{self.key}"
        x, z, th, r = polar_grid()
        path = CACHE / f"beach_bed_{self.key}.npy"
        if path.exists():
            b = np.load(path)
        else:
            b = bed_at(x, z).astype("f4")
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(path, b)
        self.bed = b
        self.waves = Waves()
        # flow: h, h*u, h*v, z. scalars: foam, wet. material: offsets of the two sets, x, z, x, z
        self.flow = [gl.target(size, dtype="f4", nearest=True) for _ in range(3)]
        self.scal = [gl.target(size, dtype="f4", nearest=False) for _ in range(2)]
        self.mat = [gl.target(size, dtype="f4", nearest=False) for _ in range(2)]
        for s in self.scal + self.mat:
            s.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bc = ctx.texture((NT, 2), 4, dtype="f4")
        self.bc.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.bc_x = R1 * np.sin(TH0 + (np.arange(NT) + .5) * D)
        self.bc_z = R1 * np.cos(TH0 + (np.arange(NT) + .5) * D)
        self.bedtex = ctx.texture(size, 1, b.tobytes(), dtype="f4")
        self.bedtex.repeat_x = self.bedtex.repeat_y = False
        self.bedtex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.n = None
        self.i, self.j = 0, 0
        self.grid = (TH0, D, RHO0, D)

    # ---- state
    def _start(self):
        """still water up to mean level, the sand damp below the berm."""
        h = np.maximum(-self.bed, 0).astype("f4")
        A = np.stack([h, 0 * h, 0 * h, self.bed], -1).astype("f4")
        wet = np.clip((1.0 - self.bed) / .5, 0, 1).astype("f4")
        B = np.stack([0 * h, wet, 0 * h, 0 * h], -1).astype("f4")
        return A, B, np.zeros_like(B)

    def _load(self, A, B, M, n):
        self.flow[self.i].tex.write(A.tobytes())
        self.scal[self.j].tex.write(B.tobytes())
        self.mat[self.j].tex.write(M.tobytes())
        self.n = n

    def _ckpt_path(self, k):
        return self.dir / f"{k:05d}.npy"

    def _save_ckpt(self, k):
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._ckpt_path(k)
        if p.exists():
            return
        a = np.frombuffer(self.flow[self.i].fbo.read(components=4, dtype="f4"), "f4")
        b = np.frombuffer(self.scal[self.j].fbo.read(components=4, dtype="f4"), "f4")
        m = np.frombuffer(self.mat[self.j].fbo.read(components=4, dtype="f4"), "f4")
        tmp = p.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.concatenate([a, b, m]).reshape(3, NR, NT, 4))
        tmp.replace(p)

    def step_of(self, t):
        # the sea outside the film's own time: earlier is the same, much later comes round again
        t = max(float(t), 0.0)
        if t > WRAP:
            t = WRAP / 2 + (t - WRAP) % (WRAP / 2)
        return int(np.floor((t + SPIN) * RATE))

    def mat_blend(self, t):
        """how much of the second set of coordinates to use, 0..1, at the time the state is at."""
        x = (self.step_of(t) / RATE / MAT_PERIOD) % 1.0
        return float(abs(2 * x - 1))

    def mat_lag(self):
        """how long ago, in seconds, the coordinates were last brought up to the water."""
        return float((self.n - 1) % SCALAR_EVERY) / RATE

    def _set_bc(self, t):
        eta, u, v = self.waves.at(self.bc_x, self.bc_z, t)
        row = np.stack([eta, u, v, 0 * eta], -1).astype("f4")
        return row

    def _flow_step(self, n):
        gl = self.gl
        t = n / RATE - SPIN + SET_AT
        dt = 1 / RATE
        A, A1, A2 = self.flow[self.i], self.flow[(self.i + 1) % 3], self.flow[(self.i + 2) % 3]
        r0 = self._set_bc(t)
        r1 = self._set_bc(t + dt)
        self.bc.write(np.stack([r0, r1]).tobytes())
        u = dict(uGrid=self.grid, uN=(NT, NR), uDt=dt, uBC=self.bc)
        gl.run("beach_swe.frag", A1, uS=A, uS0=A, uStage=0, uBCRow=(.5 / 2), **u)
        gl.run("beach_swe.frag", A2, uS=A1, uS0=A, uStage=1, uBCRow=(1.5 / 2), **u)
        self.i = (self.i + 2) % 3

    def _scalar_step(self, n):
        B, B1 = self.scal[self.j], self.scal[1 - self.j]
        self.gl.run("beach_scalar.frag", B1, uS=self.flow[self.i], uB=B, uGrid=self.grid, uN=(NT, NR), uDt=SCALAR_EVERY / RATE, uT=n / RATE)
        per = int(MAT_PERIOD * RATE)
        keep = (0.0 if n % per == 0 else 1.0, 0.0 if n % per == per // 2 else 1.0)
        self.gl.run("beach_mat.frag", self.mat[1 - self.j], uS=self.flow[self.i], uM=self.mat[self.j], uGrid=self.grid, uN=(NT, NR),
                    uDt=SCALAR_EVERY / RATE, uKeep=keep)
        self.j = 1 - self.j

    def seek(self, t):
        """bring the state to film time t."""
        target = self.step_of(t)
        if self.n == target:
            return
        K = int(CKPT * RATE)
        if self.n is None or target < self.n or target - self.n > K:
            k = target // K
            while k >= 0 and not self._ckpt_path(k).exists():
                k -= 1
            if k >= 0:
                d = np.load(self._ckpt_path(k))
                self._load(d[0], d[1], d[2], k * K)
            elif self.n is None or target < self.n:
                self._load(*self._start(), 0)
        ctx = self.gl.ctx
        t0 = time.time()
        report = target - self.n > 4 * RATE
        c = 0
        while self.n < target:
            self._flow_step(self.n)
            if self.n % SCALAR_EVERY == 0:
                self._scalar_step(self.n)
            self.n += 1
            c += 1
            if c % 64 == 0:
                ctx.finish()
            if self.n % K == 0 and self.n >= SPIN * RATE:
                self._save_ckpt(self.n // K)
            if report and self.n % (5 * RATE) == 0:
                print(f"beach sim: t={self.n / RATE - SPIN:.1f}  {time.time() - t0:.1f}s", flush=True)

    @property
    def A(self):
        return self.flow[self.i]

    @property
    def B(self):
        return self.scal[self.j]

    @property
    def M(self):
        return self.mat[self.j]

    def read(self):
        a = np.frombuffer(self.A.fbo.read(components=4, dtype="f4"), "f4").reshape(NR, NT, 4)
        b = np.frombuffer(self.B.fbo.read(components=4, dtype="f4"), "f4").reshape(NR, NT, 4)
        return a, b


def plan_view(field, xr=(-160, 260), zr=(0, 440), cell=.5):
    """resample a polar field onto a world-plane plan for debugging."""
    xs = np.arange(xr[0], xr[1], cell)
    zs = np.arange(zr[0], zr[1], cell)
    X, Z = np.meshgrid(xs, zs)
    r = np.hypot(X, Z) + 1e-3
    th = np.arctan2(X, Z)
    j = (np.log(r) - RHO0) / D - .5
    i = (th - TH0) / D - .5
    return map_coordinates(field, [j, i], order=1, mode="nearest", prefilter=False)[::-1], (xs, zs)
