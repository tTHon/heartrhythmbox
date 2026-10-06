import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# Parameters
# -----------------------------
n_surfaces = 22
n_u = 180
n_v = 360

u = np.linspace(0, 1, n_u)
v = np.linspace(0, 2 * np.pi, n_v)

U, V = np.meshgrid(u, v, indexing="ij")

# Colormap: blue -> cyan -> yellow
colors = [
    (0.01, 0.03, 0.15),
    (0.00, 0.25, 0.45),
    (0.05, 0.65, 0.70),
    (0.95, 0.90, 0.45),
    (1.00, 0.75, 0.20)
]
cmap = LinearSegmentedColormap.from_list("binary_wave", colors)

fig = plt.figure(figsize=(10, 10), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

# -----------------------------
# Create multiple curved surfaces
# -----------------------------
for k in range(n_surfaces):
    z0 = -1.0 + 2.0 * k / (n_surfaces - 1)

    # Scale changes along the vertical direction
    width = 1.05 + 0.95 * abs(z0) ** 0.75
    thickness = 0.13 + 0.05 * (1 - abs(z0))

    # Radial profile: narrow at center, broad at the outside
    radius = 0.20 + 1.05 * U ** 0.75

    # Elliptical coordinates
    X0 = width * radius * np.cos(V)
    Y0 = 0.62 * radius * np.sin(V)

    # Spiral / wave deformation
    phase = 5.5 * U + 2.2 * z0
    wave = (
        0.18 * np.sin(3 * V + phase)
        + 0.08 * np.sin(7 * V - 2 * phase)
    )

    # Make two broad lobes
    lobe = 0.28 * np.cos(2 * V) * (U ** 1.4)

    X0 = X0 + lobe * np.cos(V)
    Y0 = Y0 + lobe * np.sin(V)

    # Curvature in the third dimension
    Z0 = z0 + thickness * wave * (0.25 + U)

    # Rotate each surface slightly
    theta = 0.35 * z0 + 0.20 * np.sin(2 * z0)

    X = X0 * np.cos(theta) - Y0 * np.sin(theta)
    Y = X0 * np.sin(theta) + Y0 * np.cos(theta)
    Z = Z0

    # Color varies with height and radial position
    color_value = (
        0.45
        + 0.35 * z0
        + 0.20 * U
        + 0.08 * np.sin(2 * V)
    )

    ax.plot_surface(
        X, Y, Z,
        facecolors=cmap(np.clip(color_value, 0, 1)),
        rstride=2,
        cstride=3,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.30 + 0.025 * k
    )

# -----------------------------
# Add thin orbital / trajectory lines
# -----------------------------
t = np.linspace(0, 2 * np.pi, 1000)

for scale, angle, alpha in [
    (1.00, 0.15, 0.8),
    (1.25, -0.55, 0.5),
    (1.45, 0.80, 0.4)
]:
    x = scale * np.cos(t)
    y = 0.58 * scale * np.sin(t)
    z = 0.35 * np.sin(2 * t + angle)

    # Rotate trajectory
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = x * np.sin(angle) + y * np.cos(angle)

    ax.plot(
        xr, yr, z,
        color=(0.65, 0.85, 1.0),
        linewidth=0.45,
        alpha=alpha
    )

# -----------------------------
# Appearance
# -----------------------------
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.8, 1.8)
ax.set_zlim(-1.35, 1.35)

ax.view_init(elev=18, azim=-62)
ax.set_box_aspect((1.2, 1.0, 0.85))

ax.set_axis_off()
plt.tight_layout()
plt.show()