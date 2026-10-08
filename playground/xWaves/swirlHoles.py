"""
Crosstalk of two black holes: warped spacetime grid + swirling light rings + star field.
Two wave systems (one per hole) travel toward each other and distort the grid in between.

Usage:
    python crosstalk_bh.py            # -> crosstalk_bh.png  (1920 px wide)
    python crosstalk_bh.py --gif      # + crosstalk_bh.gif   (looping)
    python crosstalk_bh.py --sheet    # contact sheet of parameter variants
Needs: numpy, scipy, matplotlib, Pillow
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# ---------------- frame ----------------
ASPECT = 2.39            # 2.39 = cinematic scope, 16/9 = HD, 1.85 = same as the reference photo
WIDTH  = 1920
SEED   = 7

# ---------------- black holes ----------------
P1 = np.array([-0.13, -0.02])      # holes in world units (y in [-1,1], x in [-ASPECT,ASPECT])
P2 = np.array([ 0.13, 0.02])
R1, R2 = 0.10, 0.10                # horizon radii

# ---------------- crossing waves (they distort the grid) ----------------
K = 3  # wavenumber of the waves each hole sends out
M        = 2        # spiral twist of the wavefronts (integer)
WAVE_AMP = 0.05 # grid displacement from the waves
WAVE_DEC = 6 # how far the waves reach
CROSS = 1.5  # extra displacement where the two waves overlap (the crosstalk)
SWIRL = 0.5  # tangential twist of the grid around each hole
PULL = 1  # grid squeezed towards each hole

# ---------------- grid ----------------
GRID_STEP = 0.10  # cell size (world units)
GRID_ROT   = -25.0 # grid rotation (deg)
GRID_LINE  = 2.0  # line thickness (pixels)
GRID_ALPHA = 0.6  # 1 = fully black lines

# ---------------- colour of the black holes ----------------
# each hole: (base colour of halo + rings, hot colour of the brightest rings / horizon rim)  RGB 0-1
PALETTES = {
    "fire_ice": [((1.00, 0.45, 0.10), (1.00, 0.88, 0.55)),     # hole 1: orange / gold
                 ((0.10, 0.75, 1.00), (0.80, 1.00, 1.00))],    # hole 2: cyan / ice
    "fire":     [((1.00, 0.42, 0.08), (1.00, 0.85, 0.50))] * 2,
    "violet":   [((0.75, 0.25, 1.00), (1.00, 0.80, 1.00)),
                 ((1.00, 0.30, 0.55), (1.00, 0.85, 0.80))],
    "ice":      [((0.35, 0.50, 1.00), (0.85, 0.92, 1.00))] * 2,  # the original blue look
}
PALETTE     = "fire_ice"
RING_COVER  = 0.7      # how strongly rings replace the blue sky (keeps orange orange)
RING_COUPLE = 0.15     # how strongly each hole bends the other's rings (0 = independent circles)
RIM_GLOW    = 0.5      # bright rim hugging the horizon
# the disc in the middle (0 for CORE_EDGE = plain black)
CORE_EDGE   = 0.3      # brightness of the coloured disc at the horizon edge
CORE_CENTER = 0.3      # brightness at the very centre
CORE_POWER  = 5       # how quickly the colour falls off towards the centre
CORE_SWIRL  = 0.2     # faint spiral texture inside the disc

# ---------------- light rings ----------------
RING_GAIN = 0.6      # how bright the rings are (0 = invisible, 1 = full colour)
STAR_N = 300  
BLOOM     = 0.5
VIGNETTE  = 0.5
GRAIN     = 0.05
nRings    = 6           # number of rings per hole

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)

def noise2d(h, w, sigma, rng):
    n = gaussian_filter(rng.normal(size=(h, w)), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)

def make_world(width):
    w = int(width); h = int(round(width / ASPECT))
    x = np.linspace(-ASPECT, ASPECT, w)
    y = np.linspace(1, -1, h)
    X, Y = np.meshgrid(x, y)
    return X, Y, w, h

# ---------------- background: nebula + stars ----------------
def background(X, Y, w, h, rng):
    n1 = noise2d(h, w, w / 18, rng)
    n2 = noise2d(h, w, w / 60, rng)
    n3 = noise2d(h, w, w / 200, rng)
    neb = 0.55 + 0.22 * n1 + 0.10 * n2 + 0.04 * n3
    neb = np.clip(neb, 0, 1)
    c_dark = np.array([0.02, 0.03, 0.12])
    c_mid  = np.array([0.12, 0.17, 0.62])
    c_hi   = np.array([0.45, 0.50, 1.00])
    t = neb[..., None]
    img = np.where(t < 0.5, c_dark + (c_mid - c_dark) * (t / 0.5),
                            c_mid + (c_hi - c_mid) * ((t - 0.5) / 0.5))
    # a few warm dust lanes
    dust = np.clip(noise2d(h, w, w / 35, rng) - 0.8, 0, 1)
    img = img + dust[..., None] * np.array([0.10, 0.05, 0.03])
    return img

def stars(X, Y, w, h, rng):
    layer = np.zeros((h, w, 3))
    # denser towards the upper right
    prob = 0.35 + 0.65 * smoothstep(-0.2, 1.0, (X / ASPECT + Y) * 0.5 + 0.2)
    ys, xs, keep = [], [], []
    N = STAR_N * int(w / 1200 + 0.5) ** 2 if w > 1200 else STAR_N
    N = int(STAR_N * (w / 1200) ** 2)
    xi = rng.integers(0, w, N); yi = rng.integers(0, h, N)
    ok = rng.random(N) < prob[yi, xi]
    xi, yi = xi[ok], yi[ok]
    mag = rng.random(len(xi)) ** 5
    warm = rng.random(len(xi)) < 0.22
    col = np.where(warm[:, None], np.array([1.0, 0.72, 0.45]), np.array([0.85, 0.92, 1.0]))
    for c in range(3):
        np.add.at(layer[..., c], (yi, xi), (0.25 + 1.6 * mag) * col[:, c])
    small = gaussian_filter(layer, (0.8, 0.8, 0))
    # a handful of big bright stars with a soft halo
    big = np.zeros((h, w, 3))
    nb = 90
    bx = rng.integers(0, w, nb); by = rng.integers(0, h, nb)
    for c in range(3):
        np.add.at(big[..., c], (by, bx), 9.0)
    big = gaussian_filter(big, (1.8, 1.8, 0)) + 0.5 * gaussian_filter(big, (6, 6, 0))
    return small * 3.4 + big * 1.3

# ---------------- shared geometry: combined (Roche-like) radius ----------------
GRID_FOLLOW = 1.0       # 0 = grid warps around each hole as plain circles, 1 = grid follows the bent rings

def r_eff(X, Y, k):
    """Effective radius of hole k in the field of both holes: 1/r_eff = 1/r + c (R_other/R) / r_other.
    Its level sets are the bent 'rings' (stretched towards the neighbour, pinched at the saddle)."""
    holes = ((P1, R1), (P2, R2))
    P, R = holes[k]; Po, Ro = holes[1 - k]
    r = np.hypot(X - P[0], Y - P[1]) + 1e-6
    ro = np.hypot(X - Po[0], Y - Po[1]) + 1e-6
    return 1.0 / (1.0 / r + RING_COUPLE * (Ro / R) / ro), r

# ---------------- displacement of the spacetime grid ----------------
def displacement(X, Y, t):
    dx = np.zeros_like(X); dy = np.zeros_like(Y)
    waves = []
    for k, P in enumerate((P1, P2)):
        ex, ey = X - P[0], Y - P[1]
        re, r = r_eff(X, Y, k)
        rg = r + GRID_FOLLOW * (re - r)                  # radius that shapes the warp
        rx, ry = ex / r, ey / r
        th = np.arctan2(ey, ex)
        amp = np.exp(-rg / WAVE_DEC)
        wv = amp * np.cos(K * rg + M * th - t)           # wavefronts follow the bent rings
        waves.append(wv)
        dx += WAVE_AMP * wv * rx;  dy += WAVE_AMP * wv * ry          # radial wave
        sw = SWIRL * np.exp(-(rg / 0.55) ** 2)                        # twist (frame dragging)
        dx += sw * (-ry);          dy += sw * rx
        pl = PULL * np.exp(-(rg / 0.45) ** 2)                         # funnel
        dx += pl * ex;             dy += pl * ey
    # crosstalk: where the two wave systems meet they add a shear along the connecting line
    d = P2 - P1; d = d / np.linalg.norm(d); nrm = np.array([-d[1], d[0]])
    cross = waves[0] * waves[1]
    dx += CROSS * WAVE_AMP * 3 * cross * nrm[0]
    dy += CROSS * WAVE_AMP * 3 * cross * nrm[1]
    return dx, dy

def grid_lines(X, Y, w, t):
    dx, dy = displacement(X, Y, t)
    a = np.radians(GRID_ROT)
    Xw, Yw = X + dx, Y + dy
    U = (np.cos(a) * Xw - np.sin(a) * Yw) / GRID_STEP
    V = (np.sin(a) * Xw + np.cos(a) * Yw) / GRID_STEP
    out = np.zeros_like(U)
    for G in (U, V):
        gy, gx = np.gradient(G)
        width = (np.abs(gx) + np.abs(gy)) * GRID_LINE * 0.5 + 1e-6      # in grid units
        dist = np.abs(G - np.round(G))
        out = np.maximum(out, 1 - smoothstep(0.0, 1.0, dist / width))
    return out

# ---------------- swirling light rings around each hole ----------------
def rings(X, Y, w, h, t, rng_seed=3):
    rng = np.random.default_rng(rng_seed)
    light = np.zeros((h, w, 3))
    holes = ((P1, R1), (P2, R2))
    for k, (P, R) in enumerate(holes):
        Po, Ro = holes[1 - k]
        ex, ey = X - P[0], Y - P[1]
        r = np.hypot(ex, ey) + 1e-6
        th = np.arctan2(ey, ex)
        # rings follow equipotentials of BOTH holes (see r_eff)
        r_e, _ = r_eff(X, Y, k)
        rr = r_e / R                 # used by halo and rings
        rr_true = r / R              # used to keep the disc and rim attached to the hole itself
        blue, pale = (np.array(c) for c in PALETTES[PALETTE][k % 2])
        # broad coloured halo
        halo = (np.exp(-((rr - 2.1) / 1.3) ** 2) + 0.5 * np.exp(-((rr - 1.3) / 0.35) ** 2)) * (rr > 1.0)
        # concentric bright rings (random radii/strength) broken up along the angle
        ring = np.zeros_like(r)
        for j in range(nRings):
            Rj = rng.uniform(1.12, 3.6)
            sj = rng.uniform(0.015, 0.07)
            aj = rng.uniform(0.35, 1.0)
            ph = rng.uniform(0, 2 * np.pi); kk = rng.integers(1, 4)
            tw = th + 0.9 * np.log(rr + 0.1) * rng.uniform(0.5, 2.0) + 0.15 * t * (1 if k == 0 else -1)
            gate = 0.5 + 0.5 * np.cos(kk * tw + ph)
            gate = smoothstep(0.15, 0.95, gate)
            ring += aj * gate * np.exp(-((rr - Rj) / sj) ** 2)
        # fine light-trail streaks along the angular direction
        streak = gaussian_filter(rng.normal(size=(h, w)), (0.6, 0.6))
        trail = 0.55 + 0.45 * np.cos(60 * th + 6 * np.log(rr + 0.2) * 10)
        ring *= (0.55 + 0.45 * trail)
        inner = smoothstep(0.98, 1.12, rr_true)                  # rings stay outside the disc
        a = (0.85 * halo + 1.6 * ring) * inner
        rim = np.exp(-((rr_true - 1.04) / 0.07) ** 2) * (rr_true > 1.0)
        light += a[..., None] * (blue * 0.75 + pale * (0.45 * np.clip(ring[..., None] * 2, 0, 1)))
        light += (RIM_GLOW * rim)[..., None] * pale
    return light * RING_GAIN

def hole_mask(X, Y):
    m = np.ones_like(X)
    for P, R in ((P1, R1), (P2, R2)):
        r = np.hypot(X - P[0], Y - P[1])
        m *= smoothstep(R * 0.97, R * 1.06, r)
    return m

def paint_holes(img, X, Y):
    """Coloured disc: palette colour, bright at the edge and dark in the middle."""
    for k, (P, R) in enumerate(((P1, R1), (P2, R2))):
        base, hot = (np.array(c) for c in PALETTES[PALETTE][k % 2])
        ex, ey = X - P[0], Y - P[1]
        r = np.hypot(ex, ey)
        th = np.arctan2(ey, ex)
        u = np.clip(r / R, 0, 1)
        swirl = 1 + CORE_SWIRL * np.cos(2 * th + 7 * u)
        lvl = (CORE_CENTER + (CORE_EDGE - CORE_CENTER) * u ** CORE_POWER) * swirl
        core = lvl[..., None] * (0.75 * base + 0.25 * hot)
        inside = 1 - smoothstep(R * 0.97, R * 1.06, r)
        img = img * (1 - inside)[..., None] + core * inside[..., None]
    return img

# ---------------- compose ----------------
def render(t=0.0, width=WIDTH, seed=SEED):
    X, Y, w, h = make_world(width)
    rng = np.random.default_rng(seed)
    bg = background(X, Y, w, h, rng)
    st = stars(X, Y, w, h, rng)
    # stars are lensed away / dimmed close to the holes
    dim = np.ones_like(X)
    for P, R in ((P1, R1), (P2, R2)):
        dim *= 1 - 0.85 * np.exp(-(np.hypot(X - P[0], Y - P[1]) / (R * 2.4)) ** 2)
    img = bg + st * dim[..., None]
    light = rings(X, Y, w, h, t)
    cov = np.clip(light.max(axis=2) * 1.1, 0, 1)             # ring colour replaces part of the blue sky
    img = img * (1 - RING_COVER * cov)[..., None] + light
    # soft glow around the holes
    img = np.clip(img, 0, 1.6)
    # warped grid on top (dark lines)
    g = grid_lines(X, Y, w, t)
    img = img * (1 - GRID_ALPHA * g[..., None])
    # coloured holes on top
    img = paint_holes(img, X, Y)
    # cinematic finish
    img = np.clip(img, 0, 1)
    if BLOOM > 0:
        img = np.clip(img + BLOOM * gaussian_filter(np.clip(img - 0.55, 0, 1), (w / 200, w / 200, 0)), 0, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    img = img * (1 - VIGNETTE * np.clip(d - 0.45, 0, 1) ** 1.6)[..., None]
    img = np.clip(img + np.random.default_rng(1).normal(0, GRAIN, img.shape), 0, 1)
    return img

def save_static(path="playground/xWaves/crosstalk_bh.png", t=0.0, width=WIDTH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.imsave(path, render(t, width)); print("saved", path)

def save_gif(path="crosstalk_bh.gif", frames=30, width=800):
    from PIL import Image
    imgs = [Image.fromarray((render(t, width) * 255).astype(np.uint8))
            for t in np.linspace(0, 2 * np.pi, frames, endpoint=False)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0)
    print("saved", path)

if __name__ == "__main__":
    save_static()
    if "--gif" in sys.argv:
        save_gif()