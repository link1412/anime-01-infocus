"""An Iceland poppy that nobody grew.

A poppy petal is a sheet of tissue that was folded inside a bud and then let go: it has more
edge than it has room for, so the edge ruffles; the fold lines stay, radial and oblique; and
what makes it glow is that it is a tenth of a millimetre thick. Everything here follows from
that. All lengths in metres. Nothing is sampled from an image.
"""
from __future__ import annotations
import pathlib
import numpy as np
from scipy import fft as sf
from scipy import ndimage as ndi

CACHE = pathlib.Path(__file__).resolve().parents[2] / "out" / "cache"
VERSION = "flower_v1"


def white_noise(T, seed):
    return np.random.default_rng(seed).normal(size=(T, T))


def aniso(T, fphi, fr, seed):
    """gaussian-spectrum noise of unit variance. fphi cycles across a petal, fr along it (polar grid)."""
    w = white_noise(T, seed)
    fy = sf.fftfreq(T, 1 / T)[:, None]
    fx = sf.rfftfreq(T, 1 / T)[None, :]
    F = sf.rfft2(w, workers=-1) * np.exp(-(fx / fphi) ** 2 - (fy / fr) ** 2)
    n = sf.irfft2(F, s=(T, T), workers=-1)
    return ((n - n.mean()) / n.std()).astype("f4")


def ridged(n, sharp=1.0):
    return np.clip(1 - np.abs(n) / 1.35, 0, 1) ** sharp


def noise1d(n, f, seed):
    w = np.random.default_rng(seed).normal(size=n)
    k = sf.rfftfreq(n, 1 / n)
    s = sf.irfft(sf.rfft(w) * np.exp(-(k / f) ** 2), n=n)
    return (s - s.mean()) / s.std()


def creases(x, y, R, P, L, Phi, seed, n, wr, ar, lr, spread):
    """the fold lines: straight tents of finite length, as tissue keeps them."""
    T = x.shape[0]
    rng = np.random.default_rng(seed)
    h = np.zeros_like(x)
    dr, dp = L / T, 2 * Phi / T
    for _ in range(n):
        r0 = L * rng.uniform(.06, 1) ** .55
        p0 = rng.uniform(-Phi, Phi)
        ang = p0 + rng.normal(0, spread)
        w = rng.uniform(*wr)
        a = rng.uniform(*ar) * (1 if rng.random() < .62 else -1)
        l = rng.uniform(*lr)
        cx, cy = r0 * np.cos(p0), r0 * np.sin(p0)
        dx_, dy_ = np.cos(ang), np.sin(ang)
        ext = l + w
        i0 = max(int((r0 - ext) / dr), 0)
        i1 = min(int((r0 + ext) / dr) + 2, T)
        dphi = ext / max(r0 - ext, .003)
        j0 = max(int((p0 - dphi + Phi) / dp), 0)
        j1 = min(int((p0 + dphi + Phi) / dp) + 2, T)
        if i1 <= i0 or j1 <= j0:
            continue
        xx, yy = x[i0:i1, j0:j1] - cx, y[i0:i1, j0:j1] - cy
        along = xx * dx_ + yy * dy_
        across = -xx * dy_ + yy * dx_
        tent = np.clip(1 - np.abs(across) / w, 0, 1)
        win = np.clip(1 - (along / l) ** 2, 0, 1) ** 1.3
        h[i0:i1, j0:j1] += a * tent * win
    return h


def petal_maps(T, L, Phi, seed, base, tip, heart):
    """-> dict: coarse height (mesh), fine slopes, pigment. rows = r, cols = phi."""
    v = (np.arange(T) + .5) / T
    u = (np.arange(T) + .5) / T
    R, P = np.meshgrid(v * L, (u * 2 - 1) * Phi, indexing="ij")
    R, P = R.astype("f4"), P.astype("f4")
    x, y = R * np.cos(P), R * np.sin(P)
    V = R / L
    rng = np.random.default_rng(seed)

    # ---- too much edge: the margin ruffles, more and more toward the end
    ruf = np.zeros_like(R)
    for k in range(2, 7):
        a = rng.uniform(.5, 1.0) / k ** .55
        pk = rng.uniform(0, 6.28)
        bend = rng.uniform(-1.8, 1.8)
        ruf += a * np.sin(k * np.pi * P / Phi * .5 + pk + bend * V * V + .7 * np.sin(V * 3.1 + pk))
    ruf *= V ** 1.9 * .0072
    ruf += aniso(T, 6, 3.5, seed + 1) * V ** 1.4 * .0024
    # ---- the fold lines of the bud: they wander, they are not ribs
    p1 = ridged(aniso(T, 8, 3.0, seed + 2), 1.1) * .0007 * V ** .9
    p2 = ridged(aniso(T, 20, 6.5, seed + 3), 1.25) * .00020 * V ** .8
    p3 = ridged(aniso(T, 58, 17, seed + 4), 1.6) * .00011 * V ** .5
    # ---- and it was crumpled: a network of little valleys with no direction at all
    net = np.abs(aniso(T, 30, 12, seed + 15))
    net2 = np.abs(aniso(T, 110, 38, seed + 16))
    coarse = ruf + p1 + p2 + net * .00010 * V ** .6
    # ---- what stays at the scale of paper
    c1 = creases(x, y, R, P, L, Phi, seed + 5, 130, (.0016, .0048), (.00007, .00022), (.005, .016), .85)
    c2 = creases(x, y, R, P, L, Phi, seed + 6, 420, (.0006, .0016), (.00002, .00007), (.002, .008), 1.1)
    cr_mask = np.clip(np.abs(c1) * 1e3 + np.abs(c2) * 2e3, 0, 1)
    veins_a = ridged(aniso(T, 72, 2.6, seed + 7), 5.0)
    veins_b = ridged(aniso(T, 170, 3.6, seed + 8), 6.0)
    veins = np.clip(.8 * veins_a + .5 * veins_b, 0, 1)
    cells = aniso(T, 620, 620, seed + 9)
    fine = p3 + c1 + c2 + net2 * .00004 * V ** .4 + veins_a * .000012 + cells * .0000025
    gr, gp = np.gradient(fine.astype("f4"))
    dsr = L / T
    dsp = np.maximum(R, .0025) * (2 * Phi / T)
    s_p, s_r = gp / dsp, gr / dsr

    # ---- the outline
    n = T
    edge = 1 + .050 * noise1d(n, 7, seed + 10) + .009 * noise1d(n, 30, seed + 11) + .0025 * noise1d(n, 150, seed + 12)
    for _ in range(3):      # a nick or two
        c = rng.uniform(-.8, .8)
        edge -= rng.uniform(.02, .06) * np.exp(-((u * 2 - 1 - c) / rng.uniform(.006, .02)) ** 2)
    Rmax = L * (1 - .38 * np.abs(u * 2 - 1) ** 3.1) * edge
    alpha = np.clip((Rmax[None, :] - R) / .00022 + .5, 0, 1)
    alpha *= np.clip(R / .0006, 0, 1)

    # ---- pigment: how much red there is in a tenth of a millimetre
    t = np.clip(V / .85, 0, 1)[..., None]
    col = np.asarray(base, "f4") * (1 - t) + np.asarray(tip, "f4") * t
    hb = np.exp(-(V / .135) ** 2.6)[..., None]
    col = col * (1 - hb) + np.asarray(heart, "f4") * hb
    mott = aniso(T, 14, 9, seed + 13) * .045 + aniso(T, 42, 26, seed + 14) * .03
    dens = 1 + mott + .28 * hb[..., 0] - .35 * veins * 0
    thick = np.clip(dens - .10 * cr_mask + .27 * veins, .4, 2.2)
    # R,G,B = transmittance at unit thickness. A = thickness (raises the colour to this power)
    pig = np.concatenate([col, alpha[..., None]], -1).astype("f4")
    det = np.stack([s_p, s_r, cr_mask.astype("f4"), thick.astype("f4")], -1).astype("f4")
    return dict(coarse=coarse.astype("f4"), pig=pig, det=det)


def petal_surface(maps, N, L, Phi, psi0, cup, seed):
    """the mesh: a fan wrapped around a cup, displaced along its own normal."""
    T = maps["coarse"].shape[0]
    v = (np.arange(N) + .5) / N
    u = (np.arange(N) + .5) / N
    V, U = np.meshgrid(v, u, indexing="ij")
    r = V * L
    phi = (U * 2 - 1) * Phi
    # cup profile. alpha is the angle out of the flower's plane
    a0, a1, pw = cup
    ds = L / 256
    rr = (np.arange(257)) * ds
    al = a0 + (a1 - a0) * (rr / L) ** pw
    rho_t = np.cumsum(np.cos(al)) * ds
    w_t = np.cumsum(np.sin(al)) * ds
    rho = np.interp(r, rr, rho_t) + .0062
    w = np.interp(r, rr, w_t) + .0006
    c = .86 + .10 * (1 - V)
    psi = psi0 + phi * c
    B = np.stack([rho * np.cos(psi), rho * np.sin(psi), w], -1).astype("f4")
    def nrm(P):
        dv = np.gradient(P, axis=0)
        du = np.gradient(P, axis=1)
        n = np.cross(du, dv)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        return n, du, dv
    n0, _, _ = nrm(B)
    # interior of the cup is +n
    if (n0[N // 2, N // 2, 2] * 0 + (n0[N // 2, N // 2] * np.array([-np.cos(psi0), -np.sin(psi0), 0])).sum()) < 0:
        n0 = -n0
    ij = ((V * T - .5).clip(0, T - 1), (U * T - .5).clip(0, T - 1))
    h = ndi.map_coordinates(ndi.gaussian_filter(maps["coarse"], T / N * .5), ij, order=1).astype("f4")
    P = B + n0 * h[..., None]
    n, du, dv = nrm(P)
    if (n * n0).sum(-1).mean() < 0:
        n = -n
    t = du / np.maximum(np.linalg.norm(du, axis=-1, keepdims=True), 1e-12)
    b = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-12)
    uv = np.stack([U, V], -1).astype("f4")
    return dict(P=P.astype("f4"), n=n.astype("f4"), t=t.astype("f4"), b=b.astype("f4"), uv=uv, w=(V ** 2).astype("f4"))


def grid_indices(H, W, step=1):
    hh, ww = np.arange(0, H - 1, step), np.arange(0, W - 1, step)
    a, b = np.meshgrid(hh, ww, indexing="ij")
    i0 = (a * W + b).ravel()
    i1 = (a * W + np.minimum(b + step, W - 1)).ravel()
    i2 = (np.minimum(a + step, H - 1) * W + b).ravel()
    i3 = (np.minimum(a + step, H - 1) * W + np.minimum(b + step, W - 1)).ravel()
    return np.stack([i0, i1, i2, i2, i1, i3], 1).astype("u4")


def tube(path, radii, sides, uv0, uv1):
    """a tube along path (M,3). returns P,n,t,b,uv arrays shaped (M,sides,*)."""
    M = len(path)
    d = np.gradient(path, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    a = np.array([.31, .77, .55])
    s = np.cross(d, a)
    s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-9)
    c = np.cross(d, s)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=True)
    n = np.cos(th)[None, :, None] * s[:, None, :] + np.sin(th)[None, :, None] * c[:, None, :]
    P = path[:, None, :] + n * radii[:, None, None]
    t = np.cross(d[:, None, :], n)
    b = np.repeat(d[:, None, :], sides, 1)
    uu = uv0[0] + (uv1[0] - uv0[0]) * (th / (2 * np.pi))
    vv = uv0[1] + (uv1[1] - uv0[1]) * np.linspace(0, 1, M)
    uv = np.stack(np.meshgrid(vv, uu, indexing="ij"), -1)[..., ::-1]
    return P.astype("f4"), n.astype("f4"), t.astype("f4"), b.astype("f4"), uv.astype("f4")


def stamens(seed, count=86, A=1024):
    rng = np.random.default_rng(seed)
    Ps, ns, ts, bs, uvs, idx = [], [], [], [], [], []
    base_i = 0
    G = int(np.ceil(np.sqrt(count * 2)))
    cell = 1 / G
    pig = np.zeros((A, A, 4), "f4")
    pig[..., 3] = 1
    det = np.zeros((A, A, 4), "f4")
    det[..., 3] = 1

    def tile(k):
        cx, cy = k % G, k // G
        return (cx * cell + .1 * cell, cy * cell + .1 * cell), ((cx + .9) * cell, (cy + .9) * cell), cx, cy

    for k in range(count):
        th = rng.uniform(0, 6.283)
        rb = rng.uniform(.0044, .0056)
        p0 = np.array([rb * np.cos(th), rb * np.sin(th), rng.uniform(.0012, .0026)])
        L = rng.uniform(.0076, .0124)
        out = rng.uniform(.22, .78)
        p2 = p0 + L * np.array([np.cos(th) * out, np.sin(th) * out, 1.0]) / np.sqrt(1 + out * out)
        p1 = (p0 + p2) / 2 + np.array([-np.cos(th) * .0012, -np.sin(th) * .0012, .0006]) + rng.normal(0, .0005, 3)
        s = np.linspace(0, 1, 16)[:, None]
        path = (1 - s) ** 2 * p0 + 2 * s * (1 - s) * p1 + s * s * p2
        rad = np.linspace(.00022, .00011, 16)
        uv0, uv1, cx, cy = tile(2 * k)
        P, n, t, b, uv = tube(path, rad, 6, uv0, uv1)
        Ps.append(P.reshape(-1, 3)); ns.append(n.reshape(-1, 3)); ts.append(t.reshape(-1, 3)); bs.append(b.reshape(-1, 3)); uvs.append(uv.reshape(-1, 2))
        idx.append(grid_indices(16, 6) + base_i)
        base_i += 16 * 6
        x0, x1 = int(cx * cell * A), int((cx + 1) * cell * A)
        y0, y1 = int(cy * cell * A), int((cy + 1) * cell * A)
        pig[y0:y1, x0:x1, :3] = np.array([.72, .70, .30]) * rng.uniform(.8, 1.1)
        # the anther: an oblong sack of pollen on the end of it
        d = path[-1] - path[-2]
        d /= np.linalg.norm(d)
        lat = np.cross(d, rng.normal(0, 1, 3))
        lat /= np.linalg.norm(lat)
        ax = lat * .85 + d * rng.uniform(-.35, .35)
        ax /= np.linalg.norm(ax)
        la = rng.uniform(.0007, .00105)
        wa = rng.uniform(.00027, .00036)
        s2 = np.linspace(-1, 1, 12)
        pa = path[-1][None, :] + s2[:, None] * ax * la
        ra = wa * np.sqrt(np.maximum(1 - s2 * s2, .003)) * (1 + .1 * np.sin(s2 * 9 + k))
        uv0, uv1, cx, cy = tile(2 * k + 1)
        P, n, t, b, uv = tube(pa, ra, 8, uv0, uv1)
        Ps.append(P.reshape(-1, 3)); ns.append(n.reshape(-1, 3)); ts.append(t.reshape(-1, 3)); bs.append(b.reshape(-1, 3)); uvs.append(uv.reshape(-1, 2))
        idx.append(grid_indices(12, 8) + base_i)
        base_i += 12 * 8
        x0, x1 = int(cx * cell * A), int((cx + 1) * cell * A)
        y0, y1 = int(cy * cell * A), int((cy + 1) * cell * A)
        gg = rng.normal(0, 1, (y1 - y0, x1 - x0, 1))
        pig[y0:y1, x0:x1, :3] = np.array([.98, .66, .09]) * (1 + .22 * gg)
    w = np.zeros(sum(len(p) for p in Ps), "f4")
    return dict(P=np.vstack(Ps), n=np.vstack(ns), t=np.vstack(ts), b=np.vstack(bs), uv=np.vstack(uvs), w=w), np.vstack(idx), pig, det


def receptacle(seed, A=256):
    """the floor of the cup, between the stamens and where the petals begin."""
    M, K = 24, 64
    v = np.linspace(0, 1, M)
    th = np.linspace(0, 2 * np.pi, K)
    V, TH = np.meshgrid(v, th, indexing="ij")
    rho = .0090 * V
    z = .0002 + .0016 * V * V
    P = np.stack([rho * np.cos(TH), rho * np.sin(TH), z], -1)
    n = np.zeros_like(P)
    n[..., 2] = 1
    t = np.stack([-np.sin(TH), np.cos(TH), 0 * TH], -1)
    b = np.stack([np.cos(TH), np.sin(TH), 0 * TH], -1)
    uv = np.stack([TH / (2 * np.pi), V], -1)
    pig = np.zeros((A, A, 4), "f4")
    pig[..., :3] = np.array([.74, .78, .26]) * (1 + .1 * aniso(A, 40, 40, seed)[..., None])
    pig[..., 3] = 1
    det = np.zeros((A, A, 4), "f4")
    det[..., 3] = 1
    return dict(P=P.astype("f4"), n=n.astype("f4"), t=t.astype("f4"), b=b.astype("f4"), uv=uv.astype("f4"), w=np.zeros((M, K), "f4")), grid_indices(M, K), pig, det


def capsule(seed, A=512):
    M, K = 96, 96
    v = np.linspace(0, 1, M)
    th = np.linspace(0, 2 * np.pi, K)
    V, TH = np.meshgrid(v, th, indexing="ij")
    H = .0108
    Rm = .0042
    rho = Rm * (np.sin(np.clip(V * 1.16, 0, 1) * np.pi * .62) ** .62) * (1 - .12 * V)
    z = H * (V / .86).clip(0, 1) ** .9 + .001
    # the stigma: a flat star of rays on top
    top = np.clip((V - .86) / .14, 0, 1)
    rho = rho * (1 - top * top)
    ridge = np.cos(TH * 9) * .5 + .5
    z = z + top * (.0005 + .00062 * ridge ** 2 * (1 - top * .4))
    rho = rho * (1 + .035 * np.cos(TH * 9) * (1 - top) * V)
    P = np.stack([rho * np.cos(TH), rho * np.sin(TH), z], -1)
    dv = np.gradient(P, axis=0)
    du = np.gradient(P, axis=1)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    if n[M // 2, 0, 0] < 0:
        n = -n
    t = du / np.maximum(np.linalg.norm(du, axis=-1, keepdims=True), 1e-12)
    b = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-12)
    uv = np.stack([TH / (2 * np.pi), V], -1)
    pig = np.zeros((A, A, 4), "f4")
    vv = (np.arange(A) + .5) / A
    uu = (np.arange(A) + .5) / A
    VV, UU = np.meshgrid(vv, uu, indexing="ij")
    green = np.array([.42, .72, .20])
    pale = np.array([.95, .90, .45])
    k = np.clip((VV - .80) / .12, 0, 1)[..., None]
    stripes = (np.cos(UU * 2 * np.pi * 9) * .5 + .5)[..., None]
    n2 = aniso(A, 80, 80, seed)[..., None]
    pig[..., :3] = (green * (.8 + .4 * stripes) * (1 + .12 * n2)) * (1 - k) + pale * (1 + .1 * n2) * k
    pig[..., 3] = 1
    det = np.zeros((A, A, 4), "f4")
    det[..., 3] = 1
    return dict(P=P.astype("f4"), n=n.astype("f4"), t=t.astype("f4"), b=b.astype("f4"), uv=uv.astype("f4"), w=np.zeros((M, K), "f4")), grid_indices(M, K), pig, det


def stem(seed, A=256):
    """hollow, hairy, and going out of frame."""
    s = np.linspace(0, 1, 60)[:, None]
    p0 = np.array([0, 0, .0015])
    p1 = np.array([.002, -.002, -.05])
    p2 = np.array([.016, -.006, -.26])
    path = (1 - s) ** 2 * p0 + 2 * s * (1 - s) * p1 + s * s * p2
    rad = np.linspace(.0016, .00115, 60)
    rad[:5] = np.linspace(.0030, .0016, 5)
    P, n, t, b, uv = tube(path, rad, 14, (0, 0), (1, 1))
    pig = np.zeros((A, A, 4), "f4")
    pig[..., :3] = np.array([.17, .30, .085]) * (1 + .15 * aniso(A, 60, 8, seed)[..., None])
    pig[..., 3] = 1
    det = np.zeros((A, A, 4), "f4")
    det[..., 3] = 1
    return dict(P=P, n=n, t=t, b=b, uv=uv, w=np.zeros((60, 14), "f4")), grid_indices(60, 14), pig, det


def flower_transform(pos, axis, spin):
    """local (x,y in the flower's plane, z along its axis) -> world."""
    z = np.asarray(axis, "f4")
    z = z / np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = np.cos(spin), np.sin(spin)
    x, y = c * x + s * y, -s * x + c * y
    return np.stack([x, y, z], 1).astype("f4"), np.asarray(pos, "f4")


def xf_surface(S, M, o):
    out = dict(S)
    for k in ("n", "t", "b"):
        out[k] = (S[k] @ M.T).astype("f4")
    out["P"] = (S["P"] @ M.T + o).astype("f4")
    return out


# ---- the two flowers -----------------------------------------------------------------------
# orange transmittance, unit thickness
ORANGE = dict(base=(.93, .265, .012), tip=(1.0, .385, .030), heart=(.88, .72, .22))
LEMON = dict(base=(.98, .72, .040), tip=(1.0, .84, .15), heart=(.72, .80, .26))


def make_flower(seed, T, N, pal, pos, axis, spin, cups=None):
    Phi = 1.17
    L0 = .050
    out = []
    psis = [0.0, np.pi, np.pi * .5, np.pi * 1.5]
    lens = [1.0, .97, .90, .92]
    cups = cups or [(1.02, .46, 1.2), (1.05, .54, 1.2), (1.12, .62, 1.3), (1.08, .58, 1.3)]
    M, o = flower_transform(pos, axis, spin)
    for i in range(4):
        L = L0 * lens[i]
        maps = petal_maps(T, L, Phi, seed + 100 * i, pal["base"], pal["tip"], pal["heart"])
        S = petal_surface(maps, N, L, Phi, psis[i], cups[i], seed + 100 * i + 50)
        S["w"] = S["w"]
        S = xf_surface(S, M, o)
        out.append(dict(mesh=S, idx=grid_indices(N, N), lo=grid_indices(N, N, 3), pig=maps["pig"], det=maps["det"], meta=dict(L=L, Phi=Phi)))
        print(f"[flower] petal {i} seed {seed}", flush=True)
    return out, (M, o)


def extras(seed, xf):
    M, o = xf
    out = []
    for fn, step in ((stamens, 1), (capsule, 3), (stem, 2), (receptacle, 1)):
        S, idx, pig, det = fn(seed)
        S = xf_surface(S, M, o)
        out.append(dict(mesh=S, idx=idx, lo=idx, pig=pig, det=det, meta=dict(L=0.0, Phi=0.0)))
    return out


def flatten(S):
    a = [S["P"].reshape(-1, 3), S["n"].reshape(-1, 3), S["t"].reshape(-1, 3), S["b"].reshape(-1, 3), S["uv"].reshape(-1, 2), S["w"].reshape(-1, 1)]
    return np.hstack(a).astype("f4")


def downhill(pt, start, steps=600):
    """where a drop goes when it lets go: down, along the valleys, with some memory of where it was going."""
    P = pt["mesh"]["P"]
    N = P.shape[0]
    y = ndi.gaussian_filter(P[..., 1], 2.0)
    gi, gj = np.gradient(y)
    pig = pt["pig"]
    T = pig.shape[0]
    i, j = float(start[0]), float(start[1])
    vi, vj = 0.0, 0.0
    out = []
    for _ in range(steps):
        a = ndi.map_coordinates(gi, [[i], [j]], order=1)[0]
        b = ndi.map_coordinates(gj, [[i], [j]], order=1)[0]
        vi, vj = .90 * vi - a * 2.6e4 * .10, .90 * vj - b * 2.6e4 * .10
        m = np.hypot(vi, vj)
        if m > .55:
            vi, vj = vi / m * .55, vj / m * .55
        i, j = i + vi, j + vj
        out.append((i, j))
        ti, tj = int((i + .5) / N * T), int((j + .5) / N * T)
        if not (2 <= i < N - 2 and 2 <= j < N - 2) or pig[min(max(ti, 0), T - 1), min(max(tj, 0), T - 1), 3] < .5:
            break
    return np.array(out, "f4")


def sample_beads(petals, cam, seed):
    """water does not sit anywhere: it finds the valleys, and it only stays where it can."""
    rng = np.random.default_rng(seed)
    out = []
    for ip, pt in enumerate(petals):
        m = pt["mesh"]
        N = m["P"].shape[0]
        P, n, t, b = m["P"], m["n"], m["t"], m["b"]
        L, Phi = pt["meta"]["L"], pt["meta"]["Phi"]
        h = (P - ndi.gaussian_filter(P, (2.2, 2.2, 0)))
        conc = -(h * n).sum(-1)              # how far this point sits below its neighbourhood, along +n
        pig = pt["pig"]
        T = pig.shape[0]
        taken = []
        tries = 0
        target = 90
        while len(taken) < target and tries < 6000:
            tries += 1
            i, j = rng.integers(int(N * .38), N - 3), rng.integers(4, N - 4)
            ti, tj = int((i + .5) / N * T), int((j + .5) / N * T)
            if pig[ti, tj, 3] < .9:
                continue
            for sgn in (1.0, -1.0):
                nn = n[i, j] * sgn
                if nn[1] < -.15:          # it would have dropped off
                    continue
                up = nn[1]
                cv = conc[i, j] * sgn
                if rng.random() > np.clip(.25 + cv * 2200, .05, 1):
                    continue
                vis = -(nn @ ((P[i, j] - cam) / np.linalg.norm(P[i, j] - cam)))
                if vis < .22:
                    continue
                Rb = float(np.exp(rng.normal(np.log(.00062), .62)))
                Rb = min(max(Rb, .00016), .00235)
                if up < .25:
                    Rb = min(Rb, .0008)
                if any(np.linalg.norm(P[i, j] - q[0]) < (Rb + q[1]) * 1.05 for q in taken):
                    continue
                taken.append((P[i, j].copy(), Rb))
                r = (i + .5) / N * L
                dudm = 1 / (2 * Phi * max(r, .002))
                dvdm = 1 / L
                out.append(np.hstack([P[i, j], nn, t[i, j], b[i, j], [Rb, rng.uniform(1.0, 1.32), ip, (j + .5) / N, (i + .5) / N, dudm * sgn, dvdm, m["w"][i, j], rng.uniform(.10, .42)]]))
                break
    return np.array(out, "f4")


def micro_beads(petals, cam, seed):
    rng = np.random.default_rng(seed)
    out = []
    for ip, pt in enumerate(petals):
        m = pt["mesh"]
        N = m["P"].shape[0]
        T = pt["pig"].shape[0]
        for _ in range(1300):
            i, j = rng.integers(int(N * .38), N - 3), rng.integers(4, N - 4)
            ti, tj = int((i + .5) / N * T), int((j + .5) / N * T)
            if pt["pig"][ti, tj, 3] < .9:
                continue
            nn = m["n"][i, j]
            if (nn @ (m["P"][i, j] - cam)) > 0:
                nn = -nn
            if nn[1] < -.45:
                continue
            Rb = float(np.exp(rng.normal(np.log(.00011), .45)))
            r = (i + .5) / N * pt["meta"]["L"]
            out.append(np.hstack([m["P"][i, j], nn, m["t"][i, j], m["b"][i, j], [Rb, 1.0, ip, (j + .5) / N, (i + .5) / N, 1 / (2 * pt["meta"]["Phi"] * max(r, .002)), 1 / pt["meta"]["L"], m["w"][i, j], .35]]))
    return np.array(out, "f4")


def unobstructed(arr, surfaces):
    """a blurred drop half behind a sharp fold would be cut by it. keep the ones nothing stands in front of,
    and then water never has to ask what is in front of it."""
    W, H, cell = 760, 440, .001
    z = np.full((H, W), 1e9, "f4")
    for s in surfaces:
        P = s["mesh"]["P"].reshape(-1, 3)
        x = ((P[:, 0] / P[:, 2]) / cell + W / 2).astype("i4")
        y = ((P[:, 1] / P[:, 2]) / cell + H / 2).astype("i4")
        k = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        np.minimum.at(z, (y[k], x[k]), P[k, 2])
    keep = np.zeros(len(arr), bool)
    for n, a in enumerate(arr):
        c = a[:3]
        R = a[12]
        r = int(np.ceil((R * 1.3 + .0003) / c[2] / cell))
        x, y = int(c[0] / c[2] / cell + W / 2), int(c[1] / c[2] / cell + H / 2)
        win = z[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
        keep[n] = win.size > 0 and win.min() > c[2] - max(3 * R, .0025)
    return arr[keep]


HERO_POS = (-.0128, -.0043, .330)
HERO_AXIS = (-.12, .40, -.91)
SECOND_POS = (.0867, -.0325, .56)
SECOND_AXIS = (-.25, .50, -.83)
CAM = np.zeros(3, "f4")


def build():
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{VERSION}.npz"
    if fp.exists():
        d = np.load(fp, allow_pickle=True)
        return d["data"].item()
    hero, hx = make_flower(1100, 2048, 384, ORANGE, HERO_POS, HERO_AXIS, .35)
    hex_ = extras(11, hx)
    second, sx = make_flower(2300, 1024, 192, LEMON, SECOND_POS, SECOND_AXIS, 1.7)
    sex = extras(23, sx)
    beads = unobstructed(sample_beads(hero, CAM, 31), hero + hex_)
    micro = unobstructed(micro_beads(hero, CAM, 33), hero + hex_)
    beads2 = unobstructed(sample_beads(second, CAM, 37), second + sex)
    print(f"[flower] {len(beads)} beads, {len(micro)} micro, {len(beads2)} on the other one", flush=True)
    data = dict(hero=hero + hex_, second=second + sex, beads=beads, micro=micro, beads2=beads2)
    for g in ("hero", "second"):
        for s in data[g]:
            s["vb"] = flatten(s["mesh"])
            s["shape"] = s["mesh"]["P"].shape
    np.savez(fp, data=np.array(data, dtype=object))
    return data
