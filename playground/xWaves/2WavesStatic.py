"""
Crosstalk / gravitational-wave style volume render (cinematic 2.39:1).
Two spiral wave arms (m = 2) run into each other at the centre and form
nested translucent shells: red core -> yellow -> cyan -> deep blue.

Usage:
    python crosstalk_gw.py                # -> crosstalk_gw.png   (1920x804)
    python crosstalk_gw.py --gif          # + crosstalk_gw.gif    (looping)
    python crosstalk_gw.py --sheet        # parameter contact sheet
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
ASPECT  = 2.39                 # cinematic scope (use 16/9 for HD)
WIDTH   = 900
SS      = 1.25                 # supersampling (anti-aliasing for thin shells)
NZ      = 300                  # samples along the viewing ray (more = cleaner shells)

# ---------------- wave / look ----------------
L = 0.38  # half-height of the view box (smaller = zoom in)
K = 28.0  # radial wavenumber (shell spacing)
M       = 2            # spiral arms (2 = two waves meeting)
R_CORE  = 0.07
FADE = 0.70  # radial fall-off of amplitude
LEVELS  = np.array([0.35, 0.80])    # iso-levels of cos(phase) that become shells
SIGMA = 0.06  # shell thickness (small = sharp)
ALPHA   = 2.4          # overall opacity
TILT, ROLL = 58, 20  # view: tilt of orbital axis, in-plane roll (deg)
GAIN = 1.8  # exposure
COLOR_R = 0.62         # radius mapped to the blue end of the colormap
BLOOM   = 0.35         # glow strength
VIGNETTE = 0.35
GRAIN   = 0.012

CMAP = LinearSegmentedColormap.from_list("gw", [
    (0.00, "#ff3d1f"), (0.12, "#ff8a2a"), (0.28, "#ffd45a"), (0.45, "#f3f2b0"),
    (0.58, "#8fe3d6"), (0.75, "#1fa3b8"), (0.90, "#1b4fc4"), (1.00, "#10208a"),
])

def rotation(tilt, roll):
    a, b = np.radians(tilt), np.radians(roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return Rx @ Rz          # Rz = in-plane roll, Rx = tilt of the orbital axis

def field(x, y, z, t):
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    cos_t = z / r
    phi = np.arctan2(y, x)
    pol = 0.5 * (1 + cos_t ** 2)                        # quadrupole pattern
    env = pol ** 2 * np.exp(-(r / FADE) ** 2) * (r / (r + R_CORE))
    return np.cos(K * r - M * phi - t), env, r

def render(t=0.0, width=WIDTH, nz=NZ, ss=SS):
    w_px = int(width * ss)
    h_px = int(width / ASPECT * ss)
    R = rotation(TILT, ROLL)
    u = np.linspace(-L * ASPECT, L * ASPECT, w_px)
    v = np.linspace(L, -L, h_px)
    U, V = np.meshgrid(u, v)
    C = np.zeros((h_px, w_px, 3))
    T = np.ones((h_px, w_px))
    zr = 1.0                                            # depth half-range
    dw = 2 * zr / nz
    for w in np.linspace(zr, -zr, nz):                  # front-to-back
        x = R[0, 0] * U + R[0, 1] * V + R[0, 2] * w
        y = R[1, 0] * U + R[1, 1] * V + R[1, 2] * w
        z = R[2, 0] * U + R[2, 1] * V + R[2, 2] * w
        f, env, r = field(x, y, z, t)
        dens = np.zeros_like(f)
        for lv in LEVELS:
            dens += env * np.exp(-((f - lv) / SIGMA) ** 2)
        a = np.clip(1 - np.exp(-ALPHA * dens * dw * 4), 0, 1)
        col = CMAP(np.clip(r / COLOR_R, 0, 1))[..., :3]
        col = col * (0.85 + 0.35 * np.clip(dens, 0, 1))[..., None]
        C += (T * a)[..., None] * col
        T *= (1 - a)
    img = np.clip(C * GAIN, 0, 1) ** 0.9

    # downsample (anti-alias), then cinematic finish
    if ss != 1:
        hh, ww = int(h_px / ss), int(w_px / ss)
        from PIL import Image
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8))
                         .resize((int(width), int(width / ASPECT)), Image.LANCZOS)) / 255.0
    if BLOOM > 0:
        img = np.clip(img + BLOOM * gaussian_filter(img, (width / 120, width / 120, 0)), 0, 1)
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    img = img * (1 - VIGNETTE * np.clip(d - 0.35, 0, 1) ** 1.6)[..., None]
    rng = np.random.default_rng(1)
    img = np.clip(img + rng.normal(0, GRAIN, img.shape), 0, 1)
    return img

def save_static(path="playground/xWaves/crosstalk_gw.png", t=0.0):
    plt.imsave(path, render(t))
    print("saved", path)

def save_gif(path="crosstalk_gw.gif", frames=36, width=640):
    from PIL import Image
    imgs = [Image.fromarray((render(t, width, 140, 1.0) * 255).astype(np.uint8))
            for t in np.linspace(0, 2 * np.pi, frames, endpoint=False)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0)
    print("saved", path)

if __name__ == "__main__":
    save_static()
    if "--gif" in sys.argv:
        save_gif()