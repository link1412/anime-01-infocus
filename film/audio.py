"""Everything you hear is in the room, picked up by a phone's microphone.

The worlds only leak out of the camera's small speaker. A short phrase keeps stopping
one note before it ends. Only one thing in the film is allowed to be full range.

    python -m film.audio out/audio/mix.wav
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
from scipy import signal
import soundfile as sf
from .core import smooth
from .hand import Hands
from . import timeline as T

SR = 48000
DUR = T.DURATION
N = int(DUR * SR)
TT = np.arange(N) / SR
rng = np.random.default_rng(1412)
TAU = 2 * np.pi

# the phrase: it wants to fall home to D and never does
D3, A3, D4, E4, Fs4, A4, B4, D5 = 146.83, 220.0, 293.66, 329.63, 369.99, 440.0, 493.88, 587.33
PHRASE = [D4, A4, Fs4, E4]


def db(x):
    return 10 ** (x / 20)


def noise(n):
    return rng.standard_normal(n)


def pink(n):
    w = np.fft.rfft(noise(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    y = np.fft.irfft(w / np.sqrt(f), n)
    return y / np.std(y)


def sos(kind, fc, order=2):
    return signal.butter(order, fc, kind, fs=SR, output="sos")


def lp(x, fc, order=2):
    return signal.sosfilt(sos("lowpass", fc, order), x, axis=0)


def hp(x, fc, order=2):
    return signal.sosfilt(sos("highpass", fc, order), x, axis=0)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(sos("bandpass", (lo, hi), order), x, axis=0)


def env(points, t=None):
    p = np.array(points, float)
    return np.interp(TT if t is None else t, p[:, 0], p[:, 1])


def curve(fn, rate=200):
    """evaluate a film-time function at control rate, read it back at audio rate."""
    ts = np.arange(0, DUR + 1 / rate, 1 / rate)
    return np.interp(TT, ts, np.array([fn(float(t)) for t in ts], float))


def put(buf, x, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(x) <= 0:
        return
    if i < 0:
        x, i = x[-i:], 0
    m = min(len(x), len(buf) - i)
    buf[i:i + m] += x[:m] * gain


# ---- spaces ------------------------------------------------------------------------------
def _ir(rt, length, lowcut, seed):
    r = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    out = []
    for _ in range(2):
        y = r.standard_normal(n) * np.exp(-6.9 * t / rt)
        y = signal.sosfilt(sos("lowpass", lowcut), y)
        out.append(y / np.sqrt(np.sum(y * y)))
    return np.stack(out, 1)


IR_ROOM = _ir(.34, .5, 5200, 7)
IR_HALL = _ir(2.6, 3.4, 7000, 8)


def in_space(x, ir, wet=.28):
    """mono in, stereo out."""
    d = np.stack([x, x], 1)
    w = np.stack([signal.fftconvolve(x, ir[:, c])[:len(x)] for c in (0, 1)], 1)
    return d * (1 - wet) + w * wet


SPK_BP = sos("bandpass", (620, 5000), 2)
SPK_PK_B, SPK_PK_A = signal.iirpeak(2100, 1.1, fs=SR)


def spk(x):
    """a speaker the size of a fingernail."""
    y = signal.sosfilt(SPK_BP, x)
    y = y + .7 * signal.lfilter(SPK_PK_B, SPK_PK_A, y)
    return np.tanh(y * 1.8) / 1.8


# ---- small sounds ---------------------------------------------------------------------------
def key(strength=1.0, heavy=False):
    n = int(.11 * SR)
    t = np.arange(n) / SR
    click = bp(noise(n), 1500, 7200) * np.exp(-t / .0014)
    f0 = rng.uniform(170, 250) * (.8 if heavy else 1.0)
    thock = np.sin(TAU * f0 * t) * np.exp(-t / (.02 if heavy else .012)) * .55
    thock += bp(noise(n), 220, 1300) * np.exp(-t / .007) * .5
    y = click * .8 + thock
    up = int(rng.uniform(.045, .075) * SR)
    if up < n:
        y[up:] += (click[:n - up] * .28)
    return y * strength * rng.uniform(.8, 1.1)


def tick(f=3100, dur=.009, a=1.0):
    n = int(dur * 4 * SR)
    t = np.arange(n) / SR
    return a * np.sin(TAU * f * t) * np.exp(-t / dur)


def bell(f, dur=2.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for r, a, tau in ((1, 1, .9), (2.0, .55, .55), (3.01, .3, .35), (4.21, .22, .2), (5.43, .12, .12)):
        y += a * np.sin(TAU * f * 2 * r * t) * np.exp(-t / tau)
    return y * (1 - np.exp(-t / .0015))


def pluck(f, dur=2.4, decay=.996, bend=0.0):
    n = int(dur * SR)
    p = max(int(round(SR / f)), 2)
    x = np.zeros(n)
    x[:p] = lp(noise(p), 3800)
    a = np.zeros(p + 2)
    a[0], a[p], a[p + 1] = 1, -decay * .5, -decay * .5
    y = signal.lfilter([1.0], a, x)
    if bend:
        # a finger sliding on the string as the note thins out
        t = np.arange(n) / SR
        rate = 1 + bend * smooth(.35, .9, t) * (1 + .4 * np.sin(TAU * 4.6 * t))
        y = np.interp(np.cumsum(rate), np.arange(n), y)
    return y


def chip(f, dur, duty=.25, a=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.where(np.mod(f * t, 1.0) < duty, 1.0, -1.0)
    e = np.floor(np.clip(1 - t / max(dur * 1.05, 1e-3), 0, 1) * 8) / 8
    return lp(y * e * a, 7500)


def piano(f, dur, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    B = .00035
    for h in range(1, 15):
        fh = h * f * np.sqrt(1 + B * h * h)
        if fh > 15000:
            break
        tau = 3.4 * (440 / f) ** .25 / (1 + .32 * h)
        a = 1 / h ** 1.2
        for det in (-.00014, .00014):
            y += a * .5 * np.sin(TAU * fh * (1 + det) * t + rng.uniform(0, TAU)) * np.exp(-t / tau)
    return y * (1 - np.exp(-t / .004)) * vel


# ---- the room ------------------------------------------------------------------------------
print("room", flush=True)
air = np.stack([lp(hp(pink(N), 90), 4200) for _ in range(2)], 1) * db(-60)
hiss = np.stack([hp(noise(N), 2500) for _ in range(2)], 1) * db(-73)

f0 = 118 + 1.2 * np.sin(TAU * TT / 17) + .6 * np.sin(TAU * TT / 5.3)
ph = np.cumsum(TAU * f0 / SR)
fan = sum(a * np.sin(h * ph + i) for i, (h, a) in enumerate(((1, 1), (2, .6), (3, .35), (4, .5), (5, .2), (6, .16), (8, .12))))
fan = fan * (1 + .08 * np.sin(TAU * TT / 3.1)) * db(-62) + bp(pink(N), 220, 2400) * db(-61)
fan = in_space(fan, IR_ROOM, .55)


def make_rain():
    out = []
    for c in range(2):
        y = np.zeros(N)
        kinds = 10
        for _ in range(kinds):
            imp = np.zeros(N)
            count = int(520 * DUR / kinds)
            np.add.at(imp, rng.integers(0, N, count), rng.lognormal(-1.2, .75, count))
            k = int(rng.uniform(.0035, .0085) * SR)
            ker = noise(k) * np.exp(-np.arange(k) / SR / rng.uniform(.0007, .0017))
            y += signal.oaconvolve(imp, ker)[:N]
        y = bp(y, 900, 6400)
        bed = bp(pink(N), 350, 4600) * (1 + .22 * np.sin(TAU * TT / 7.1 + c) + .12 * np.sin(TAU * TT / 2.3 + 2 * c))
        y = y / np.std(y) * db(-47) + bed * db(-53)
        # the odd heavy drop from the frame
        for t in rng.uniform(0, DUR, 40):
            n = int(.05 * SR)
            d = np.sin(TAU * rng.uniform(600, 1300) * np.arange(n) / SR) * np.exp(-np.arange(n) / SR / .009)
            put(y, d, t, db(-41) * rng.uniform(.4, 1))
        out.append(y)
    return np.stack(out, 1)


rain = make_rain()

# ---- hands, breath ----------------------------------------------------------------------------
print("body", flush=True)
hands = Hands(DUR, tremor=T.Film._tremor, breath=T.Film._breath)
ts = np.arange(0, DUR, 1 / 200)
hm = np.array([[hands.at(float(t))[k] for k in ("yaw", "pitch", "roll")] for t in ts])
speed = np.linalg.norm(np.gradient(hm, 1 / 200, axis=0), axis=1)
speed = speed / (np.percentile(speed, 95) + 1e-9)
rub = lp(noise(N), 800) * np.interp(TT, ts, np.clip(speed, 0, 3)) ** 1.3 * db(-55)

breath_amt = curve(T.Film._breath)
phi = TAU * .24 * TT
ex = np.clip(np.sin(phi), 0, 1) ** 2
inh = np.clip(-np.sin(phi), 0, 1) ** 2
br = bp(noise(N), 320, 1700) * ex + bp(noise(N), 900, 3800) * inh * .55
breath_body = br * db(-57) * breath_amt

# ---- autofocus motor --------------------------------------------------------------------------


def focus_at(t):
    if t < T.T_SEA:
        return T.FOCUS(t)
    if T.T_WORDS <= t < T.T_STOP:
        return T.W_FOCUS(t - T.T_WORDS)
    if t >= T.T_BACK:
        return T.END_FOCUS_RACK(t)
    return 0.0


fd = curve(focus_at)
fs_ = np.abs(np.gradient(signal.sosfiltfilt(sos("lowpass", 30), fd), 1 / SR))
tr = bp(noise(N), 1300, 4200) * .45 + np.sign(np.sin(np.cumsum(TAU * (110 + 170 * np.clip(fs_, 0, 2)) / SR))) * .5
motor = bp(tr, 500, 5000) * np.clip(fs_, 0, 1.6) * db(-52)

# ---- what is struck in the room / what leaks from the speaker ----------------------------------
print("events", flush=True)
room_hits = np.zeros(N)     # keys, buttons, the hand on the body
leak = np.zeros(N)          # everything that comes out of the camera

# magnifier clicks: a small dial in the hand and the camera's own tick
for t0, _, _ in T.MAG_STEPS + T.STOP_IN + T.STOP_OUT:
    put(room_hits, key(.35), t0 - .02, db(-30))
    put(leak, tick(), t0, db(-26))

# the second drop: it lands, and much later it goes over the edge and takes up its weight
def plink(f, dur, rise, a=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return a * np.sin(np.cumsum(TAU * f * (1 + rise * t / dur)) / SR) * np.exp(-t / (dur * .26)) * (1 - np.exp(-t / .0007))


put(leak, plink(980, .045, .35), T.T_LAND, db(-33))
put(leak, plink(540, .10, .85), T.T_OVER + .04, db(-34))
put(leak, plink(760, .06, .55), T.T_OVER + .19, db(-40))

# the sea, heard through the bead
sea_t = env([(11.45, 0), (12.75, .2), (13.8, .45), (T.T_SEA, 1), (T.T_MENU, .9), (T.T_INK - .3, .9), (T.T_INK + .3, 0)])
swell = .55 + .3 * np.sin(TAU * TT / 6.3) + .15 * np.sin(TAU * TT / 2.7 + 1)
leak += lp(pink(N), 2600) * swell * sea_t * db(-30)
leak += bp(noise(N), 2000, 6000) * np.clip(swell - .55, 0, 1) ** 1.5 * sea_t * db(-30)
for t0 in (T.T_SEA + 3.2, T.T_SEA + 3.55):         # a far gull
    n = int(.26 * SR)
    t = np.arange(n) / SR
    f = 2300 - 900 * t / .26
    put(leak, np.sin(np.cumsum(TAU * f / SR)) * np.sin(np.pi * t / .26) ** 2, t0, db(-34))
for i, f in enumerate(PHRASE):
    put(leak, bell(f), T.T_SEA + 1.75 + i * .62, db(-30))

# the whip, and whatever it is that cracks at the edge of that sea
yaw_v = curve(lambda t: abs(T.SEA_YAW(t + .01) - T.SEA_YAW(t - .01)) / .02 if T.T_SEA <= t < T.T_MENU else 0.0)
leak += bp(noise(N), 500, 4000) * np.clip(yaw_v / 2.5, 0, 1) ** 1.4 * db(-26)
n = int(.25 * SR)
thump = np.sin(TAU * 170 * np.arange(n) / SR) * np.exp(-np.arange(n) / SR / .05) + lp(noise(n), 500) * np.exp(-np.arange(n) / SR / .03)
put(room_hits, thump, T.T_WHIP - .03, db(-26))
for i in range(7):
    put(leak, bp(noise(600), 3000, 8500) * np.exp(-np.arange(600) / SR / .002), T.T_WHIP + .22 + i * rng.uniform(.012, .05), db(-26) * .8 ** i)

# the menu
put(room_hits, key(.6), T.T_MENU - .06, db(-28))
for i in range(11):
    tt = T.T_MENU + .35 + i * .15
    put(room_hits, key(.3), tt - .015, db(-34))
    put(leak, tick() if i < 10 else tick(1250, .03, 1.5), tt, db(-26))

# ink
ink_t = env([(T.T_INK - .2, 0), (T.T_INK + .7, 1), (T.T_WORDS - .4, 1), (T.T_WORDS, 0)])
pan_v = curve(lambda t: abs(T.INK_PAN(t + .01) - T.INK_PAN(t - .01)) / .02 if T.T_INK <= t < T.T_WORDS else 0.0)
leak += bp(pink(N), 700, 3400) * np.clip(pan_v, 0, 2) ** .9 * db(-30)
leak += bp(pink(N), 1000, 3000) * ink_t * db(-50)
for i, f in enumerate(PHRASE):
    put(leak, pluck(f / 2, 3.0, .9965, bend=.012 if i == 3 else 0.0), T.T_INK + 2.45 + i * .72, db(-26))
# words: the wall falls as a rain of tiny ticks, then the sentence lands on a real keyboard
for _ in range(90):
    put(leak, tick(rng.uniform(2200, 4200), .004), T.T_WORDS + rng.uniform(.9, 2.6), db(-40) * rng.uniform(.3, 1))
for k in range(sum(len(s) for s in T.SENTENCE)):
    put(room_hits, key(.55 + .3 * rng.random()), T.T_WORDS + 5.0 + k * .06 + .85, db(-29))

# pixel: a square wave that also gives up one note before home, and stops dead the moment it's noticed
P = T.T_PIXEL
for i, f in enumerate(PHRASE):
    d = .4 if i < 3 else 3.0 - 1.4 - 3 * .4
    put(leak, chip(f * 2, d, .25), P + 1.4 + i * .4, db(-34))
for i in range(int(1.6 / .2)):
    put(leak, chip(D3 if i % 2 == 0 else A3, .19, .5), P + 1.4 + i * .2, db(-36))
for i in range(int(1.6 / .1)):
    put(leak, lp(noise(int(.03 * SR)), 9000) * np.linspace(1, 0, int(.03 * SR)), P + 1.4 + i * .1, db(-42))
for i, f in enumerate((A4, B4)):
    put(leak, chip(f * 2, .06, .5), P + 3.6 + i * .07, db(-34))
wave = bp(noise(N), 1500, 6000) * (.5 + .5 * np.floor(np.mod(TT * 8, 2))) * env([(P + .4, 0), (P + 1.3, 1), (P + 4.6, 1), (P + 5.8, 0)])
leak += wave * db(-44)

# back at the desk
D = T.T_DESK
for i in range(27):
    put(room_hits, key(.8), D + 2.2 + i / 22 + rng.normal(0, .006), db(-27))
put(room_hits, key(1.2, heavy=True), T.T_LIVE - .15, db(-23))

# ---- the tunnel: the speaker's output goes back into the microphone ------------------------------
print("tunnel", flush=True)
x = np.zeros(N)
seq = [(bell, D4), (pluck, A4 / 2), (chip, Fs4 * 2), (bell, E4), (pluck, A4 / 2), (chip, E4 * 2)]
k = 0
for t0 in np.arange(T.T_LIVE + .1, T.T_WHITE, 8 / 30):
    fn, f = seq[k % len(seq)]
    y = fn(f, .5) if fn is not chip else chip(f, .22)
    put(x, y, t0, db(-26))
    k += 1
x += (air[:, 0] + fan[:, 0] * 2) * 3
g = env([(T.T_LIVE, .6), (T.T_LIVE + 1.2, 1.0), (T.T_LIVE + 3, 1.35), (T.T_WHITE, 1.75)])
loop = sos("bandpass", (340, 1700), 2)
d = int(8 / 30 * SR)
y = np.zeros(N)
B = 1024
zi = signal.sosfilt_zi(loop) * 0
i0, i1 = int(T.T_LIVE * SR), int((T.T_WHITE + .4) * SR)
for i in range(i0, i1, B):
    j = min(i + B, N)
    fb, zi = signal.sosfilt(loop, y[i - d:j - d], zi=zi)
    y[i:j] = x[i:j] + np.tanh(fb * g[i:j] * 1.5) / 1.5
howl = spk(y) * env([(T.T_LIVE - .1, 0), (T.T_LIVE + .1, 1)])

# ---- the end ------------------------------------------------------------------------------------
print("end", flush=True)
for name, off in T.REALITY_OFF.items():
    put(room_hits, key(.9), off - .02, db(-27))
put(room_hits, key(.4), T.T_BACK, db(-34))

# ---- the only thing with a body -------------------------------------------------------------------
print("chord", flush=True)
R = T.RETURN + .11
chord_m = np.zeros(N)
notes = ((73.42, 1.1), (110.0, .7), (146.83, .8), (220.0, .55), (293.66, 1.0), (369.99, .6), (440.0, .5), (587.33, .55))
# the chord from the first cut, note for note: its partials start at the phases they started at then, whatever else has drawn from the generator since
_film_rng = rng
rng = np.random.Generator(np.random.PCG64())
rng.bit_generator.state = dict(bit_generator="PCG64", state=dict(state=269663066119132239393381759157598024962, inc=299786216803941571686981521372269601319), has_uint32=0, uinteger=0)
for f, v in notes:
    put(chord_m, piano(f, 6.0, v), R, 1.0)
rng = _film_rng
chord = in_space(chord_m, IR_HALL, .42) * db(-14)
pad_t = TT - R
# the pad and the sub are counted from the strike: on the film's clock their low notes beat against the piano's differently
# whenever the film changes length, and the chord comes out up to 3 dB louder or quieter
T_CH = pad_t + 86.36
pad = np.zeros((N, 2))
pe = smooth(0.0, .55, pad_t) * (pad_t >= 0)
for c, det in ((0, -.0011), (1, .0011)):
    for f, v in ((146.83, 1.0), (220.0, .8), (293.66, .7), (369.99, .45), (440.0, .4)):
        for h in range(1, 7):
            pad[:, c] += v / h ** 1.7 * np.sin(TAU * f * h * (1 + det) * T_CH + .3 * h * np.sin(TAU * .17 * T_CH))
pad *= (pe * db(-37))[:, None]
sub = (np.sin(TAU * 73.42 * T_CH) * smooth(0.0, .18, pad_t) * (pad_t >= 0) * db(-23))[:, None] * np.array([[1.0, 1.0]])
# the note that has been hanging since the sea: it was never finished, only held
held = np.sin(TAU * E4 * TT) + .3 * np.sin(TAU * E4 * 2 * TT)
held = held * env([(T.E + 19.5, 0), (R - .15, 1), (R + .55, 0)]) * db(-34)
held = np.stack([held, held], 1)

# ---- mix ----------------------------------------------------------------------------------------
print("mix", flush=True)
room_gate = curve(lambda t: T.reality("room", t))[:, None]
grain_gate = curve(lambda t: T.reality("grain", t))[:, None]
tremor_gate = curve(lambda t: T.reality("tremor", t))
rain_lvl = env([(0, 1.0), (T.T_SEA, .72), (T.T_DESK, .42), (T.T_END, .42)])[:, None]
fan_lvl = env([(0, 0.9), (T.T_DESK, 1.0), (T.T_DESK + 2.5, 1.5), (T.T_END, 1.5)])[:, None]

# a phone's microphone has no low end at all
mic = air + fan * fan_lvl + rain * rain_lvl
mic = mic + in_space(room_hits, IR_ROOM, .34)
mic = mic + in_space(spk(leak), IR_ROOM, .16)
mic = mic + np.stack([rub * tremor_gate, rub * tremor_gate], 1) * .9
mic = mic + np.stack([motor, motor], 1)
mic = lp(hp(mic, 150), 9000)
mic = (mic * room_gate + hiss * grain_gate) * db(10)
mic = mic + in_space(howl, IR_ROOM, .2) * db(-7)

# the tunnel takes everything over; then the phone's AGC climbs back out of the white
over = env([(T.T_WHITE + .399, 1), (T.T_WHITE + .4, 0), (T.T_WHITE + .7, 0), (T.T_BACK, 1)])[:, None]
mic = mic * over

# breath is not in the table. it is mixed outside the gate
bm = lp(hp(breath_body, 150), 9000) * db(6)
mic = mic + np.stack([bm, bm], 1)

# the camera's own focus beep. the room isn't back yet: it comes straight out of the silence
beep = np.zeros(N)
for t0 in (T.RETURN, R):
    n_ = int(.075 * SR)
    t_ = np.arange(n_) / SR
    put(beep, np.sin(TAU * 2730 * t_) * np.minimum(t_ / .002, 1) * np.minimum((.075 - t_) / .006, 1), t0, db(-27))
beep = np.stack([beep, beep], 1)

# the picture is gone and so is the room. only the chord gets out, and it is let die on its own
mic = mic * (TT < T.T_END)[:, None]
tail = env([(T.T_END + 2.6, 1), (T.DURATION - .25, 0)])[:, None]
# a phone's speaker can't reach the bottom of this and doesn't need the top of it: the quiet rooms come up, the two loud moments go down.
# the beeps stay where they were, and the chord comes down a little to meet what is around it
ride = env([(0, 3), (T.T_SEA - 1, 3), (T.T_SEA + 2, 0), (T.T_INK + 9, 0), (T.T_WORDS - .5, 3), (T.T_STOP, 3), (T.T_PIXEL + 1, 0),
            (T.T_DESK, 0), (T.T_DESK + 1, 3), (T.T_LIVE + 2, 3), (T.T_WHITE - 1.5, -1.5), (T.T_WHITE + .6, -1.5), (T.T_BACK + .5, 4), (T.T_END, 4)])
mix = mic * db(ride)[:, None] + held + (chord + pad + sub) * tail * db(-2) + beep
mix = np.tanh(mix * 1.08) / 1.08 * .92

# exact digital silence where the film says so, and after the cut
t_off = T.REALITY_OFF["room"] + .10
quiet = (TT >= t_off) & (TT < T.T_CREEP)
mix[quiet] = 0.0

# ---- out -----------------------------------------------------------------------------------------
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/audio/mix.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, np.vstack([mix, np.zeros((int(T.FILM_END * SR) - len(mix), 2))]), SR, subtype="PCM_24")   # the card is silent

# what the camera's meters read, at the film's frame rate
fps = 30
hop = SR // fps
fr = len(mix) // hop
lev = np.zeros((fr, 2))
hold = np.array([0.0, 0.0])
for i in range(fr):
    seg = mix[i * hop:(i + 1) * hop]
    p = np.abs(seg).max(axis=0)
    v = np.clip((20 * np.log10(np.maximum(p, 1e-6)) + 60) / 60, 0, 1)
    v[p <= 0] = 0.0
    hold = np.maximum(v, hold - 20 / 60 / fps)
    hold[p <= 0] = 0.0
    lev[i] = hold
np.save(out.parent / "levels.npy", lev)
print("wrote", out, "peak", float(np.abs(mix).max()), flush=True)
