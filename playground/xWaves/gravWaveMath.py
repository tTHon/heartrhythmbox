import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# SETTINGS
# =========================================================

n_layers = 34
n_u = 220
n_v = 420

# ความกว้างรวมของรูปทรง
main_width = 2.10
main_height = 1.35

# ความแรงของการบิดเป็นสองปีก
lobe_strength = 0.72

# ความแรงของการบิดรอบแกนกลาง
twist_strength = 1.25

# ความโค้งตามแกน z
vertical_wave = 0.42

# มุมกล้อง
camera_elev = 18
camera_azim = -62

# =========================================================
# GRID
# =========================================================

u = np.linspace(0, 1, n_u)
v = np.linspace(0, 2 * np.pi, n_v)

U, V = np.meshgrid(u, v, indexing="ij")

# =========================================================
# COLOR MAP
# =========================================================

colors = [
    (0.005, 0.015, 0.06),
    (0.00, 0.12, 0.32),
    (0.00, 0.42, 0.58),
    (0.10, 0.70, 0.70),
    (0.86, 0.88, 0.57),
    (1.00, 0.72, 0.20)
]

cmap = LinearSegmentedColormap.from_list(
    "reference_colors",
    colors
)

# =========================================================
# ROTATION
# =========================================================

def rotate_xy(x, y, angle):
    xr = x * np.cos(angle) - y * np.sin(angle)
    yr = x * np.sin(angle) + y * np.cos(angle)
    return xr, yr


# =========================================================
# CREATE ONE BIPOLAR WAVE SURFACE
# =========================================================

def make_bipolar_surface(layer_index):

    # ตำแหน่งแนวดิ่งของชั้นนี้
    z_layer = (
        -1.0
        + 2.0 * layer_index / (n_layers - 1)
    )

    # ความเข้มของชั้นบริเวณด้านนอก
    radial = 0.18 + 0.82 * U ** 0.72

    # มุมบิดเพิ่มขึ้นจากด้านในออกด้านนอก
    twist = (
        twist_strength
        * (U ** 1.25)
        * np.sin(2 * V)
    )

    # -----------------------------------------------------
    # สร้างรูปทรงพื้นฐาน
    # -----------------------------------------------------

    # ความกว้างในแนว x
    # cos(2V) ทำให้เกิดปีกใหญ่สองข้าง
    two_lobes = (
        1.0
        + lobe_strength
        * np.cos(2 * V)
        * (0.25 + 0.75 * U)
    )

    # ความกว้างหลัก
    x_radius = (
        main_width
        * radial
        * two_lobes
    )

    # ความสูงในแนว y
    y_radius = (
        main_height
        * radial
        * (
            0.78
            + 0.22 * np.cos(2 * V)
        )
    )

    # มุมที่ใช้หมุน cross-section
    angle = (
        V
        + twist
        + 0.30 * z_layer
    )

    # รูปร่าง elliptical shell
    X = x_radius * np.cos(angle)
    Y = y_radius * np.sin(angle)

    # -----------------------------------------------------
    # ทำให้ตรงกลางแคบและด้านนอกแผ่กว้าง
    # -----------------------------------------------------

    center_pinching = (
        0.35
        + 0.65 * U
    )

    X = X * center_pinching
    Y = Y * center_pinching

    # -----------------------------------------------------
    # สร้างการบิดตัวในแนว z
    # -----------------------------------------------------

    Z = (
        z_layer
        + vertical_wave
        * radial
        * np.sin(2 * V + 1.2 * U)
    )

    # เพิ่มการยกตัวบริเวณปีก
    Z = Z + (
        0.13
        * np.cos(2 * V)
        * U ** 1.4
    )

    # -----------------------------------------------------
    # หมุนทั้งชั้นเล็กน้อย
    # -----------------------------------------------------

    layer_angle = (
        0.38 * z_layer
    )

    X, Y = rotate_xy(
        X,
        Y,
        layer_angle
    )

    return X, Y, Z


# =========================================================
# FIGURE
# =========================================================

fig = plt.figure(
    figsize=(10, 10),
    facecolor="black"
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.set_facecolor("black")

# =========================================================
# DRAW LAYERS
# =========================================================

for layer_index in range(n_layers):

    X, Y, Z = make_bipolar_surface(layer_index)

    layer = (
        -1.0
        + 2.0 * layer_index / (n_layers - 1)
    )

    # สีเหลืองบริเวณกลางและสีฟ้าบริเวณด้านนอก
    color_value = np.clip(
        0.52
        + 0.28 * U
        - 0.18 * abs(layer),
        0,
        1
    )

    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=cmap(color_value),
        rcount=100,
        ccount=180,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.23
    )


# =========================================================
# CENTRAL SMALL SPIRAL
# =========================================================

uc = np.linspace(0, 1, 100)
vc = np.linspace(0, 2 * np.pi, 260)

UC, VC = np.meshgrid(uc, vc, indexing="ij")

central_r = 0.08 + 0.42 * UC

central_angle = (
    VC
    + 2.0 * np.pi * 0.75 * UC
)

Xc = (
    central_r
    * np.cos(central_angle)
)

Yc = (
    0.70
    * central_r
    * np.sin(VC)
)

Zc = (
    0.24
    * np.sin(2 * central_angle)
    * (1 - UC)
)

ax.plot_surface(
    Xc,
    Yc,
    Zc,
    color=(1.0, 0.75, 0.25),
    linewidth=0,
    alpha=0.28,
    shade=True
)


# =========================================================
# LONG THIN LINES
# =========================================================

t = np.linspace(-1, 1, 800)

for offset in [-0.16, 0.0, 0.16]:

    # เส้นแนวเฉียงผ่านจุดกลาง
    x = 0.55 * t + offset
    y = 0.05 * np.sin(4 * t)
    z = 1.25 * t

    ax.plot(
        x,
        y,
        z,
        color=(0.72, 0.88, 1.0),
        linewidth=0.45,
        alpha=0.55
    )


# =========================================================
# CAMERA AND DISPLAY
# =========================================================

ax.view_init(
    elev=camera_elev,
    azim=camera_azim
)

ax.set_xlim(-2.50, 2.50)
ax.set_ylim(-1.85, 1.85)
ax.set_zlim(-1.40, 1.40)

ax.set_box_aspect(
    (1.45, 1.00, 0.85)
)

ax.set_axis_off()

plt.tight_layout()
plt.show()

# บันทึกไฟล์ได้ด้วยคำสั่งนี้
# plt.savefig(
#     "closer_reference_shape.png",
#     dpi=300,
#     facecolor="black",
#     bbox_inches="tight"
# )