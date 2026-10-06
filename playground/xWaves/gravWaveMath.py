import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LightSource

# =========================================================
# 1) PARAMETERS
# =========================================================

# Resolution
n_layers = 28
n_u = 190
n_v = 360

# ---------------------------------------------------------
# Two-wave geometry
# ---------------------------------------------------------


# Position of each wave center
left_center = -0.85
right_center = 0.85

# Width of each wave
left_width = 1.30
right_width = 1.15

# Height / thickness in y direction
left_height = 0.95
right_height = 0.90

# Rotation of each wave in degrees
left_angle_deg = 20
right_angle_deg = -20

left_angle = np.deg2rad(left_angle_deg)
right_angle = np.deg2rad(right_angle_deg)

# ---------------------------------------------------------
# Curl parameters
# ---------------------------------------------------------

# Main curling strength
# Higher value = more curled
left_curl = 5.0
right_curl = 5.0

# Additional twist between layers
left_twist = 0.85
right_twist = -0.85

# Wave thickness along z
layer_spacing = 1.90

# Strength of wave undulation
wave_strength = 0.22

# Number of turns around the center
n_turns = 1.55


# =========================================================
# 2) PARAMETER GRID
# =========================================================

# u controls movement from center toward the outer edge
u = np.linspace(0.02, 1.0, n_u)

# v controls the cross-section of each ribbon
v = np.linspace(-np.pi, np.pi, n_v)

U, V = np.meshgrid(u, v, indexing="ij")

# =========================================================
# 3) COLOR MAP
# =========================================================

colors = [
    (0.005, 0.015, 0.07),
    (0.00, 0.12, 0.35),
    (0.00, 0.42, 0.62),
    (0.20, 0.72, 0.70),
    (0.92, 0.88, 0.52),
    (1.00, 0.70, 0.18)
]

wave_cmap = LinearSegmentedColormap.from_list(
    "curled_wave",
    colors
)

# =========================================================
# 4) ROTATION FUNCTION
# =========================================================

def rotate_xy(X, Y, angle):
    """
    Rotate coordinates around the z-axis.
    """

    X_new = (
        X * np.cos(angle)
        - Y * np.sin(angle)
    )

    Y_new = (
        X * np.sin(angle)
        + Y * np.cos(angle)
    )

    return X_new, Y_new


# =========================================================
# 5) CREATE ONE CURLED WAVE
# =========================================================

def create_curled_wave(
    center_x,
    direction,
    layer,
    width,
    height,
    curl,
    twist,
    phase_shift,
    rotation_angle
):
    """
    Create a curled ribbon-like wave.

    center_x:
        Center position of the wave.

    direction:
        +1 for left wave moving right.
        -1 for right wave moving left.

    layer:
        z-position of the surface layer.

    width:
        Overall radial width.

    height:
        Thickness across the ribbon.

    curl:
        Curling strength.

    twist:
        Rotation change between layers.

    phase_shift:
        Phase difference between left and right wave.

    rotation_angle:
        Overall orientation angle.
    """

    # -----------------------------------------------------
    # Radial distance from the inner center
    # -----------------------------------------------------

    # Nonlinear profile:
    # small near the center, broad at the outside
    radius = (
        0.08
        + width
        * (
            0.18 * U
            + 0.82 * U ** 0.72
        )
    )

    # -----------------------------------------------------
    # Angle around the spiral
    # -----------------------------------------------------

    # Several turns as the surface moves outward
    spiral_angle = (
        direction
        * (
            2.0 * np.pi
            * n_turns
            * U
        )
        + V
        + phase_shift
        + twist * layer
    )

    # -----------------------------------------------------
    # Elliptical cross-section
    # -----------------------------------------------------

    # Width perpendicular to the spiral
    cross_width = (
        0.10
        + height
        * (
            0.18
            + 0.82 * U
        )
    )

    # Main curled coordinates
    X = radius * np.cos(spiral_angle)
    Y = cross_width * np.sin(V)

    # Move the two wave centers apart
    X = direction * X + center_x

    # -----------------------------------------------------
    # Add wave-like deformation
    # -----------------------------------------------------

    deformation_phase = (
        curl * U
        + 2.0 * V
        + phase_shift
    )

    deformation = (
        np.sin(deformation_phase)
        + 0.35
        * np.sin(
            3.0 * deformation_phase
            - 2.0 * V
        )
    )

    # Deformation is stronger toward the outside
    deformation *= (
        wave_strength
        * (0.25 + 0.90 * U)
    )

    # Apply deformation perpendicular to the spiral
    X = X + (
        deformation
        * np.cos(spiral_angle)
    )

    Y = Y + (
        0.75
        * deformation
        * np.sin(spiral_angle)
    )

    # -----------------------------------------------------
    # z curvature
    # -----------------------------------------------------

    # Two broad lobes, similar to the reference image
    z_wave = (
        0.20 * np.sin(2.0 * spiral_angle)
        + 0.10 * np.sin(5.0 * spiral_angle)
    )

    Z = (
        layer
        + z_wave * (0.25 + 0.75 * U)
        + 0.08 * deformation
    )

    # -----------------------------------------------------
    # Rotate the complete wave
    # -----------------------------------------------------

    X, Y = rotate_xy(
        X,
        Y,
        rotation_angle
    )

    return X, Y, Z


# =========================================================
# 6) CREATE FIGURE
# =========================================================

fig = plt.figure(
    figsize=(11, 11),
    facecolor="black"
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.set_facecolor("black")

# =========================================================
# 7) DRAW THE TWO CURLED WAVES
# =========================================================

for layer_index in range(n_layers):

    # z-position of each translucent layer
    layer = (
        -layer_spacing / 2
        + layer_spacing
        * layer_index
        / (n_layers - 1)
    )

    # -----------------------------------------------------
    # Left curled wave
    # -----------------------------------------------------

    X_left, Y_left, Z_left = create_curled_wave(
        center_x=left_center,
        direction=+1,
        layer=layer,
        width=left_width,
        height=left_height,
        curl=left_curl,
        twist=left_twist,
        phase_shift=0.0,
        rotation_angle=left_angle
    )

    # -----------------------------------------------------
    # Right curled wave
    # -----------------------------------------------------

    X_right, Y_right, Z_right = create_curled_wave(
        center_x=right_center,
        direction=-1,
        layer=layer,
        width=right_width,
        height=right_height,
        curl=right_curl,
        twist=right_twist,
        phase_shift=np.pi,
        rotation_angle=right_angle
    )

    # -----------------------------------------------------
    # Colors
    # -----------------------------------------------------

    left_color_value = np.clip(
        0.20
        + 0.45 * U
        + 0.18 * layer,
        0,
        1
    )

    right_color_value = np.clip(
        0.38
        + 0.35 * U
        - 0.12 * layer,
        0,
        1
    )

    # -----------------------------------------------------
    # Draw left surface
    # -----------------------------------------------------

    ax.plot_surface(
        X_left,
        Y_left,
        Z_left,
        facecolors=wave_cmap(left_color_value),
        rcount=80,
        ccount=150,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.25
    )

    # -----------------------------------------------------
    # Draw right surface
    # -----------------------------------------------------

    ax.plot_surface(
        X_right,
        Y_right,
        Z_right,
        facecolors=wave_cmap(right_color_value),
        rcount=80,
        ccount=150,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.25
    )


# =========================================================
# 8) CENTRAL COLLISION REGION
# =========================================================

# A translucent central rotating region
central_u = np.linspace(0.02, 1.0, 120)
central_v = np.linspace(0, 2 * np.pi, 240)

CU, CV = np.meshgrid(
    central_u,
    central_v,
    indexing="ij"
)

central_radius = 0.05 + 0.48 * CU

central_angle = (
    2.0 * np.pi * 1.6 * CU
    + CV
)

X_center = (
    central_radius
    * np.cos(central_angle)
)

Y_center = (
    0.42
    * central_radius
    * np.sin(CV)
)

Z_center = (
    0.32
    * np.sin(2.0 * central_angle)
    * (1.0 - CU)
)

X_center, Y_center = rotate_xy(
    X_center,
    Y_center,
    0.0
)

ax.plot_surface(
    X_center,
    Y_center,
    Z_center,
    color=(1.0, 0.72, 0.20),
    linewidth=0,
    antialiased=True,
    alpha=0.28,
    shade=True
)


# =========================================================
# 9) THIN TRAJECTORY LINES
# =========================================================

t = np.linspace(0, 1, 700)

for side in [-1, 1]:

    for line_index in range(5):

        if side == -1:
            start_x = -2.0
            end_x = -0.05
            line_angle = left_angle
        else:
            start_x = 2.0
            end_x = 0.05
            line_angle = right_angle

        x = (
            start_x * (1 - t)
            + end_x * t
        )

        y = (
            0.22
            * np.sin(
                2.0 * np.pi * t
                + line_index * 0.8
            )
            * (1.0 - t)
        )

        z = (
            0.42
            * np.sin(
                2.7 * np.pi * t
                + line_index
            )
        )

        x, y = rotate_xy(
            x,
            y,
            line_angle
        )

        ax.plot(
            x,
            y,
            z,
            color=(0.72, 0.90, 1.0),
            linewidth=0.40,
            alpha=0.40
        )


# =========================================================
# 10) CAMERA AND APPEARANCE
# =========================================================

# Camera angle
ax.view_init(
    elev=20,
    azim=-58
)

# Axis limits
ax.set_xlim(-2.35, 2.35)
ax.set_ylim(-1.80, 1.80)
ax.set_zlim(-1.35, 1.35)

# Aspect ratio
ax.set_box_aspect(
    (1.45, 1.00, 0.85)
)

# Hide axes
ax.set_axis_off()

plt.tight_layout()

# Display
plt.show()

# Save image if needed:
# plt.savefig(
#     "two_strongly_curled_waves.png",
#     dpi=300,
#     facecolor="black",
#     bbox_inches="tight"
# )