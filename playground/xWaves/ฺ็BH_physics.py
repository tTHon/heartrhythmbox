"""
Crosstalk of two black holes: warped spacetime grid + swirling light rings + star field.

Grid physics (GRID_MODEL = "physics"):
  * Near field  - Brill-Lindquist two-hole initial data: spatial metric psi^4 * delta_ij with
                  psi = 1 + m1/(2 r1) + m2/(2 r2). Grid lines are drawn at equal *proper* spacing, so they
                  crowd towards each hole (the two funnels overlap = the "crosstalk").
  * Far field   - leading-order gravitational wave of a circular binary (quadrupole formula, TT gauge).
                  For an observer in the orbital plane h_phiphi = (G A / c^4 r) cos(2(phi - Omega t_ret)) / 2,
                  A = 2 mu a^2 Omega^2. Accumulated circumference stretch gives the tangential grid shift
                  xi_phi = (G A / 4 c^4) sin(2 phi - 2 Omega t + k r),  k = 2 Omega / c = 2 pi / lambda_GW,
                  with amplitude xi0 = G^2 m1 m2 / (2 c^4 a) (from Kepler's law). The holes really orbit.
  Exaggerated / regularised: strain amplitude (EXAG), near-zone taper, funnel strength cap.
  The light rings, stars and nebula are decorative.

Usage:
    python crosstalk_bh.py            # -> crosstalk_bh.png  (1920 px wide)
    python crosstalk_bh.py --gif      # + crosstalk_bh.gif   (looping)
    python crosstalk_bh.py --sheet    # contact sheet of parameter variants
Needs: numpy, scipy, matplotlib, Pillow
"""
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

# ---------------- black holes (G = c = 1 units) ----------------
# Horizon radius (isotropic coordinates) of an isolated Schwarzschild hole is m/2.
MASSES = (0.40, 0.34)              # m1, m2  (world units)
P1_0   = np.array([-0.62,  0.22])  # positions at t = 0 (y in [-1,1], x in [-ASPECT,ASPECT])
P2_0   = np.array([ 0.36, -0.30])

# ---------------- grid model ----------------
GRID_MODEL = "physics"   # "physics" = GW strain + Brill-Lindquist funnel;  "decor" = old hand-made warp
# --- far-field gravitational wave of a circular binary (quadrupole formula, TT gauge) ---
LAMBDA = 0.55            # GW wavelength drawn (world units). None -> real Kepler value (about 4 screen heights)
EXAG   = 1.8             # strain exaggeration (real h ~ 1e-21 is invisible)
FAR_DECAY = None         # None = physical (displacement amplitude constant in r); e.g. 2.0 = fade for looks
# --- near-field: spatial metric  gamma_ij = psi^4 delta_ij,  psi = 1 + sum m_i / (2 r_i) ---
FUNNEL_FAR = 2.0         # radius where the funnel map is blended to zero (regularisation)
FUNNEL     = 1.0         # 0 = off, 1 = full (capped so the map never folds over)
# --- old decorative terms (used only when GRID_MODEL == "decor") ---
K        = 8.5
M        = 1
WAVE_AMP = 0.115
WAVE_DEC = 2.2
CROSS    = 0.8
SWIRL    = 0.22
PULL     = 0.75

# ---------------- grid ----------------
GRID_STEP = 0.088  # cell size (world units)
GRID_ROT   = -14.0 # grid rotation (deg)
GRID_LINE  = 1.25   # line thickness (pixels)
GRID_ALPHA = 0.85  # 1 = fully black lines

# ---------------- light rings ----------------
RING_GAIN = 1.0
STAR_N = 16000  
BLOOM     = 0.30
VIGNETTE  = 0.35
GRAIN     = 0.010

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

# ---------------- binary: orbit ----------------
def binary_state(alpha):
    """Return list of (position, horizon radius, mass) with the pair rotated by `alpha` about the COM."""
    m1, m2 = MASSES
    com = (m1 * P1_0 + m2 * P2_0) / (m1 + m2)
    c, s_ = np.cos(alpha), np.sin(alpha)
    Rm = np.array([[c, -s_], [s_, c]])
    p1 = com + Rm @ (P1_0 - com)
    p2 = com + Rm @ (P2_0 - com)
    return [(p1, m1 / 2, m1), (p2, m2 / 2, m2)], com

def orbit_constants():
    m1, m2 = MASSES
    Mt = m1 + m2
    a = np.linalg.norm(P2_0 - P1_0)
    Om = np.sqrt(Mt / a ** 3)                      # Kepler, G = 1
    xi0 = m1 * m2 / (2 * a)                         # G^2 m1 m2 / (2 c^4 a)
    k = 2 * Om if LAMBDA is None else 2 * np.pi / LAMBDA
    return a, Om, xi0, k

def alpha0():
    d = P2_0 - P1_0
    return np.arctan2(d[1], d[0])

def smoothstep_inv(a, b, x):
    return 1 - smoothstep(a, b, x)

def displacement_physics(X, Y, alpha, holes, com):
    a, Om, xi0, k = orbit_constants()
    dx = np.zeros_like(X); dy = np.zeros_like(Y)
    # ---- far field: tangential shift from the quadrupole GW ----
    ex, ey = X - com[0], Y - com[1]
    r = np.hypot(ex, ey) + 1e-6
    phi = np.arctan2(ey, ex)
    xi = EXAG * xi0 * np.sin(2 * phi - 2 * (alpha + alpha0()) + k * r)
    xi *= 1 - np.exp(-(k * r / 1.5) ** 2)           # near-zone taper (formula invalid for k r << 1)
    if FAR_DECAY is not None:
        xi *= np.exp(-r / FAR_DECAY)
    dx += -xi * ey / r
    dy += xi * ex / r
    # ---- near field: equal-proper-spacing map around each hole (Brill-Lindquist) ----
    if FUNNEL > 0:
        for P, R, m in holes:
            ex, ey = X - P[0], Y - P[1]
            rr = np.hypot(ex, ey) + 1e-6
            Rf = FUNNEL_FAR
            rc = np.maximum(rr, R * 1.0)
            # e(r) = l(r) - l(Rf) - (r - Rf) with l(r) = r + m ln r - m^2/(4 r)  (isotropic Schwarzschild)
            e = m * np.log(rc / Rf) - 0.25 * m * m * (1 / rc - 1 / Rf)
            # cap the strength so r + s*e stays > 0 down to the horizon
            eh = m * np.log(R / Rf) - 0.25 * m * m * (1 / R - 1 / Rf)
            s_cap = min(1.0, 0.9 * R / abs(eh)) if eh < 0 else 1.0
            e = FUNNEL * s_cap * e * smoothstep_inv(0.55 * Rf, Rf, rr)
            dx += e * ex / rr
            dy += e * ey / rr
    return dx, dy

def displacement_decor(X, Y, t, holes):
    dx = np.zeros_like(X); dy = np.zeros_like(Y)
    waves = []
    for P, R, m in holes:
        ex, ey = X - P[0], Y - P[1]
        r = np.hypot(ex, ey) + 1e-6
        rx, ry = ex / r, ey / r
        th = np.arctan2(ey, ex)
        wv = np.exp(-r / WAVE_DEC) * np.cos(K * r + M * th - t)
        waves.append(wv)
        dx += WAVE_AMP * wv * rx;  dy += WAVE_AMP * wv * ry
        sw = SWIRL * np.exp(-(r / 0.55) ** 2)
        dx += sw * (-ry);          dy += sw * rx
        pl = PULL * np.exp(-(r / 0.45) ** 2)
        dx += pl * ex;             dy += pl * ey
    d = holes[1][0] - holes[0][0]; d = d / np.linalg.norm(d); nrm = np.array([-d[1], d[0]])
    cross = waves[0] * waves[1]
    dx += CROSS * WAVE_AMP * 3 * cross * nrm[0]
    dy += CROSS * WAVE_AMP * 3 * cross * nrm[1]
    return dx, dy

def grid_lines(X, Y, w, alpha, holes, com):
    if GRID_MODEL == "physics":
        dx, dy = displacement_physics(X, Y, alpha, holes, com)
    else:
        dx, dy = displacement_decor(X, Y, alpha, holes)
    a = np.radians(GRID_ROT)
    Xw, Yw = X + dx, Y + dy
    U = (np.cos(a) * Xw - np.sin(a) * Yw) / GRID_STEP
    V = (np.sin(a) * Xw + np.cos(a) * Yw) / GRID_STEP
    out = np.zeros_like(U)
    for G in (U, V):
        gy, gx = np.gradient(G)
        width = (np.abs(gx) + np.abs(gy)) * GRID_LINE * 0.5 + 1e-6
        dist = np.abs(G - np.round(G))
        out = np.maximum(out, 1 - smoothstep(0.0, 1.0, dist / width))
    return out

# ---------------- swirling light rings around each hole ----------------
def rings(X, Y, w, h, t, holes, rng_seed=3):
    rng = np.random.default_rng(rng_seed)
    light = np.zeros((h, w, 3))
    pale = np.array([0.85, 0.92, 1.0]); blue = np.array([0.35, 0.50, 1.0])
    for k, (P, R, _m) in enumerate(holes):
        ex, ey = X - P[0], Y - P[1]
        r = np.hypot(ex, ey) + 1e-6
        th = np.arctan2(ey, ex)
        rr = r / R
        # broad blue halo
        halo = (np.exp(-((rr - 2.1) / 1.3) ** 2) + 0.5 * np.exp(-((rr - 1.3) / 0.35) ** 2)) * (rr > 1.0)
        # concentric bright rings (random radii/strength) broken up along the angle
        ring = np.zeros_like(r)
        for j in range(14):
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
        inner = smoothstep(0.98, 1.12, rr)                       # keep the hole black
        a = (0.85 * halo + 1.6 * ring) * inner
        light += a[..., None] * (blue * 0.55 + pale * (0.45 * np.clip(ring[..., None] * 2, 0, 1)))
    return light * RING_GAIN

def hole_mask(X, Y, holes):
    m = np.ones_like(X)
    for P, R, _m in holes:
        r = np.hypot(X - P[0], Y - P[1])
        m *= smoothstep(R * 0.97, R * 1.06, r)
    return m

# ---------------- compose ----------------
def render(t=0.0, width=WIDTH, seed=SEED):
    """t = orbital phase (radians). The holes rotate by t about the centre of mass."""
    X, Y, w, h = make_world(width)
    rng = np.random.default_rng(seed)
    holes, com = binary_state(t)
    bg = background(X, Y, w, h, rng)
    st = stars(X, Y, w, h, rng)
    dim = np.ones_like(X)
    for P, R, _m in holes:
        dim *= 1 - 0.85 * np.exp(-(np.hypot(X - P[0], Y - P[1]) / (R * 2.4)) ** 2)
    img = bg + st * dim[..., None]
    img = img + rings(X, Y, w, h, t, holes)
    img = np.clip(img, 0, 1.6)
    g = grid_lines(X, Y, w, t, holes, com)
    img = img * (1 - GRID_ALPHA * g[..., None])
    img = img * hole_mask(X, Y, holes)[..., None]
    img = np.clip(img, 0, 1)
    if BLOOM > 0:
        img = np.clip(img + BLOOM * gaussian_filter(np.clip(img - 0.55, 0, 1), (w / 200, w / 200, 0)), 0, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    img = img * (1 - VIGNETTE * np.clip(d - 0.45, 0, 1) ** 1.6)[..., None]
    img = np.clip(img + np.random.default_rng(1).normal(0, GRAIN, img.shape), 0, 1)
    return img

def save_static(path="playground/xWaves/crosstalk_bh.png", t=0.0, width=WIDTH):
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