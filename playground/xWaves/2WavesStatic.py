"""
Crosstalk / gravitational-wave style volume render.
Two spiral wave arms (m = 2) run into each other at the centre and form
nested translucent shells: red core -> yellow -> cyan -> deep blue.

Usage:
    python crosstalk_gw.py            # -> crosstalk_gw.png
    python crosstalk_gw.py --gif      # + crosstalk_gw.gif (looping)
Needs: numpy, matplotlib, Pillow (for --gif)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------- parameters ----------------
SIZE    = 900          # output pixels (square)
NZ      = 220          # samples along the viewing ray
L       = 0.75         # half-size of the view box (smaller = zoom in)
K       = 20.0         # radial wavenumber  (shell spacing)
M       = 2            # spiral arms (2 = two waves meeting, quadrupole)
R_CORE  = 0.07         # softening radius at the centre
FADE    = 0.80         # radial fall-off of amplitude
LEVELS  = np.array([0.30, 0.75])               # iso-levels of cos(phase) that become shells
SIGMA   = 0.11         # shell thickness
ALPHA   = 2.6          # overall opacity
TILT, ROLL = 58, -12   # view angles (degrees): tilt of orbital axis, in-plane roll
GAIN    = 1.3          # exposure
COLOR_R = 0.75         # radius that maps to the blue end of the colormap
SHOW_AXES = False      # the two thin white lines in the reference image

# colour by radius: core -> outskirts
CMAP = LinearSegmentedColormap.from_list("gw", [
    (0.00, "#ff3d1f"),   # red core
    (0.12, "#ff8a2a"),   # orange
    (0.28, "#ffd45a"),   # yellow
    (0.45, "#f3f2b0"),   # pale yellow-white
    (0.58, "#8fe3d6"),   # cyan
    (0.75, "#1fa3b8"),   # teal
    (0.90, "#1b4fc4"),   # blue
    (1.00, "#10208a"),   # deep blue
])

def rotation(tilt, roll):
    a, b = np.radians(tilt), np.radians(roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return Rx @ Rz   # Rz = in-plane roll of the image, Rx = tilt of the orbital axis

def field(x, y, z, t):
    """Normalised wave cos(phase), amplitude envelope, radius (orbital axis = z)."""
    r = np.sqrt(x * x + y * y + z * z) + 1e-9
    cos_t = z / r
    phi = np.arctan2(y, x)
    pol = 0.5 * (1 + cos_t ** 2)                       # quadrupole pattern
    env = pol ** 2 * np.exp(-(r / FADE) ** 2) * (r / (r + R_CORE))
    return np.cos(K * r - M * phi - t), env, r

def render(t=0.0, size=SIZE, nz=NZ):
    R = rotation(TILT, ROLL)
    u = np.linspace(-L, L, size)
    U, V = np.meshgrid(u, -u)
    C = np.zeros((size, size, 3))
    T = np.ones((size, size))
    dw = 2 * L / nz
    for w in np.linspace(L, -L, nz):                   # front-to-back
        x = R[0, 0] * U + R[0, 1] * V + R[0, 2] * w
        y = R[1, 0] * U + R[1, 1] * V + R[1, 2] * w
        z = R[2, 0] * U + R[2, 1] * V + R[2, 2] * w
        f, env, r = field(x, y, z, t)
        # opacity peaks where the field crosses the chosen iso-levels -> shells
        dens = np.zeros_like(f)
        for lv in LEVELS:
            dens += env * np.exp(-((f - lv) / SIGMA) ** 2)
        a = np.clip(1 - np.exp(-ALPHA * dens * dw * 4), 0, 1)
        col = CMAP(np.clip(r / COLOR_R, 0, 1))[..., :3]
        # slight brightening on shell edges for a glassy look
        col = col * (0.85 + 0.35 * np.clip(dens, 0, 1))[..., None]
        C += (T * a)[..., None] * col
        T *= (1 - a)
    img = np.clip(C * GAIN, 0, 1) ** 0.9
    if SHOW_AXES:
        for (x0, y0, x1, y1) in [(0.63, 0.0, 0.64, 1.0), (0.23, 1.0, 0.78, 0.0)]:
            n = size * 2
            xs = (np.linspace(x0, x1, n) * (size - 1)).astype(int)
            ys = (np.linspace(y0, y1, n) * (size - 1)).astype(int)
            img[ys.clip(0, size - 1), xs.clip(0, size - 1)] = 1
    return img

def save_static(path="playground/xWaves/crosstalk_gw.png", t=0.0):
    plt.imsave(path, render(t))
    print("saved", path)

def save_gif(path="playground/xWaves/crosstalk_gw.gif", frames=36, size=420):
    from PIL import Image
    imgs = [Image.fromarray((render(t, size, 90) * 255).astype(np.uint8))
            for t in np.linspace(0, 2 * np.pi, frames, endpoint=False)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=70, loop=0)
    print("saved", path)

if __name__ == "__main__":
    save_static()
    if "--gif" in sys.argv:
        save_gif()