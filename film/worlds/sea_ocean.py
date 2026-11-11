"""Tessendorf FFT ocean, evaluated exactly at any t (no state, nothing to loop).

Five cascades of one sea. Each owns one band of wavelengths, so together they are one
continuous JONSWAP spectrum (plus a long swell) from hundreds of metres down to
centimetres. Periods are incommensurate and every tile is rotated by its own angle,
which is what keeps the repeat invisible.

Texture layout per cascade (all world-frame, metres / dimensionless slopes):
  A = (Dx, h, Dz, sx*sz)
  B = (sx, sz, sx*sx, sz*sz)
Storing the second moments means hardware mip + anisotropic filtering gives us the true
slope distribution of a pixel footprint (LEAN mapping): the mean is the resolved normal,
the covariance is the unresolved roughness that drives the sun glitter.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np

G = 9.81
KC = 370.0          # gravity-capillary crossover wavenumber

# period L, grid N, longest and shortest wavelength owned, tile rotation (deg), choppiness
CASCADES = [
    (2039.0, 512, 1e9, 15.0, 17.0, .8),
    (433.0, 512, 15.0, 3.4, -31.0, .78),
    (97.3, 512, 3.4, .77, 41.0, .62),
    (21.7, 512, .77, .175, -11.0, .45),
    (4.93, 512, .175, .04, 23.0, .3),
]


def _omega(k):
    return np.sqrt(G * k * (1 + (k / KC) ** 2))


def _spread(theta, s):
    """cos^2s(theta/2) directional spreading, normalised over the circle."""
    from scipy.special import gammaln
    ln_norm = (2 * s - 1) * np.log(2.0) + 2 * gammaln(s + 1) - np.log(np.pi) - gammaln(2 * s + 1)
    return np.exp(ln_norm) * np.abs(np.cos(theta / 2)) ** (2 * s)


def _spectrum(kx, kz, k):
    """Directional wavenumber spectrum Psi(kx,kz), m^4. Integral over the plane is the height variance."""
    out = np.zeros_like(k)
    ok = k > 1e-6
    kk = k[ok]
    w = _omega(kk)
    dwdk = G * (1 + 3 * (kk / KC) ** 2) / (2 * w)
    ang = np.arctan2(kx[ok], kz[ok])     # angle of k from +z (towards the sun), positive to the right

    # wind sea: JONSWAP, waves running towards the camera and slightly to its left
    alpha, gamma = .0105, 2.2
    wp = np.sqrt(G * 2 * np.pi / 27.0)
    sig = np.where(w <= wp, .07, .09)
    r = np.exp(-(w - wp) ** 2 / (2 * sig ** 2 * wp ** 2))
    S = alpha * G * G * w ** -5 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** r
    dir_wind = np.pi - .42
    s = np.where(w < wp, 9.0 * (w / wp) ** 4.0, 9.0 * (w / wp) ** -2.2)
    s = np.clip(s, .9, 40.0)
    psi = S * dwdk * _spread(ang - dir_wind, s) / kk

    # long swell from another storm, a slow breathing under everything
    ws = 2 * np.pi / 10.5
    Hs = .5
    Ss = (Hs / 4) ** 2 * np.exp(-.5 * ((w - ws) / .035) ** 2) / (.035 * np.sqrt(2 * np.pi))
    psi += Ss * dwdk * _spread(ang - (np.pi + .62), 26.0) / kk
    out[ok] = psi
    return out


class Cascade:
    def __init__(self, L, N, lam_hi, lam_lo, rot_deg, chop, seed):
        self.L, self.N, self.chop = L, N, chop
        self.rot = np.radians(rot_deg)
        c, s = np.cos(self.rot), np.sin(self.rot)
        n = np.fft.fftfreq(N, 1 / N)
        ktx, ktz = np.meshgrid(n * 2 * np.pi / L, n * 2 * np.pi / L)   # array index [z, x]
        kx = c * ktx - s * ktz
        kz = s * ktx + c * ktz
        k = np.hypot(kx, kz)
        band = (k >= 2 * np.pi / lam_hi) & (k < 2 * np.pi / lam_lo)
        psi = _spectrum(kx, kz, k) * band
        rng = np.random.default_rng(seed)
        dk = 2 * np.pi / L
        xi = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2)
        h0 = xi * np.sqrt(psi / 2) * dk
        idx = (-np.arange(N)) % N
        h0m = np.conj(h0[np.ix_(idx, idx)])
        self.w = _omega(np.maximum(k, 1e-6)).astype("f8")
        self.h0, self.h0m = h0 * N * N, h0m * N * N       # numpy's ifft2 divides by N^2
        safe = np.maximum(k, 1e-6)
        self.kx, self.kz = kx, kz
        self.dxk, self.dzk = -1j * kx / safe, -1j * kz / safe
        self.dxx, self.dzz, self.dxz = kx * kx / safe, kz * kz / safe, kx * kz / safe
        self.var_h = float(np.sum(psi) * dk * dk)
        self.var_s = float(np.sum(psi * k * k) * dk * dk)

    def at(self, t):
        e = np.exp(1j * self.w * t)
        hk = self.h0 * e + self.h0m * np.conj(e)
        ik_x, ik_z = 1j * self.kx, 1j * self.kz
        a = np.fft.ifft2(hk + 1j * (self.dxk * hk))
        b = np.fft.ifft2(self.dzk * hk + 1j * (ik_x * hk))
        c = np.fft.ifft2(ik_z * hk + 1j * (self.dxx * hk))
        d = np.fft.ifft2(self.dzz * hk + 1j * (self.dxz * hk))
        h, dx = a.real, a.imag
        dz, sx = b.real, b.imag
        sz, dxx = c.real, c.imag
        dzz, dxz = d.real, d.imag
        lam = self.chop
        jxx, jzz, jxz = 1 + lam * dxx, 1 + lam * dzz, lam * dxz
        det = np.maximum(jxx * jzz - jxz * jxz, .34)
        # true slopes of the displaced (Gerstner-like) surface
        psx = (sx * jzz - sz * jxz) / det
        psz = (sz * jxx - sx * jxz) / det
        A = np.stack([lam * dx, h, lam * dz, psx * psz], -1)
        B = np.stack([psx, psz, psx * psx, psz * psz], -1)
        return A.astype("f2"), B.astype("f2")


class Ocean:
    def __init__(self, seed=20261104):
        rngs = np.random.SeedSequence(seed).spawn(len(CASCADES))
        self.cascades = [Cascade(*c, int(r.generate_state(1)[0])) for c, r in zip(CASCADES, rngs)]
        self.pool = ThreadPoolExecutor(len(CASCADES))
        self.sigma_h = float(np.sqrt(sum(c.var_h for c in self.cascades)))
        self.var_s = float(sum(c.var_s for c in self.cascades))

    def at(self, t):
        res = list(self.pool.map(lambda c: c.at(t), self.cascades))
        return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])

    def describe(self):
        rows = [f"L={c.L:7.1f}  sigma_h={np.sqrt(c.var_h):.3f} m  var_slope={c.var_s:.4f}" for c in self.cascades]
        rows.append(f"total sigma_h={self.sigma_h:.3f} m (Hs={4 * self.sigma_h:.2f} m)  var_slope={self.var_s:.4f}")
        return "\n".join(rows)
