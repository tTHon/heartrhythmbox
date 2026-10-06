import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# CROSS TALK — ACADEMIC SESSION BACKGROUND (INTERFERENCE)
# ============================================================

frames = 240
FPS = 30

x_lim = 15
y_lim = 8.4375  # 16:9 Aspect Ratio

x = np.linspace(-x_lim, x_lim, 800)
y = np.linspace(-y_lim, y_lim, 450)
X, Y = np.meshgrid(x, y)

# ============================================================
# FIGURE SETUP
# ============================================================
fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

bg_color = "#030712"
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

ax.set_xlim(-x_lim, x_lim)
ax.set_ylim(-y_lim, y_lim)
ax.set_aspect("equal")
ax.axis("off")

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Background Glow
background = np.exp(-((X / 12) ** 2) - ((Y / 6) ** 2))
ax.imshow(
    background,
    extent=[-x_lim, x_lim, -y_lim, y_lim],
    origin="lower",
    cmap="viridis",
    alpha=0.15,
    aspect="auto",
)

# ============================================================
# WAVE PARAMETERS
# ============================================================
n_rings = 20
ring_spacing = 1.1
wave_speed = 0.05
max_radius = 32.0

left_source_x = -x_lim
right_source_x = x_lim

# พื้นที่เว้นสำหรับข้อความตรงกลาง
center_mask_w = 9.0
center_mask_h = 3.5

# ============================================================
# CREATE RINGS & INTERFERENCE POINTS
# ============================================================
left_rings = []
right_rings = []

for i in range(n_rings):
    (l,) = ax.plot([], [], color="#38bdf8", linewidth=0.9, alpha=0)
    (r,) = ax.plot([], [], color="#818cf8", linewidth=0.9, alpha=0)
    left_rings.append(l)
    right_rings.append(r)

# จุดสว่างจากการแทรกสอด (Interference Nodes)
(interf_dots,) = ax.plot(
    [], [], "o", color="#f43f5e", markersize=3, alpha=0, zorder=5
)
(interf_glow,) = ax.plot(
    [], [], "o", color="#fb7185", markersize=7, alpha=0, zorder=4
)

# ============================================================
# TEXT ELEMENTS
# ============================================================
title_text = "C R O S S   T A L K"

ax.text(
    0, 0.4,
    title_text,
    color="#f8fafc",
    fontsize=32,
    fontweight="bold",
    ha="center",
    va="center",
    alpha=0.95
)

ax.text(
    0, -0.6,
    "ACADEMIC SESSION",
    color="#94a3b8",
    fontsize=13,
    ha="center",
    va="center",
    alpha=0.75
)

# ============================================================
# WAVE & INTERFERENCE LOGIC
# ============================================================
theta = np.linspace(0, 2 * np.pi, 400)

def get_ring_data(cx, radius):
    if radius <= 0:
        return None, None
    xx = cx + radius * np.cos(theta)
    yy = radius * np.sin(theta)
    return xx, yy

def mask_center(xx, yy):
    in_center_x = (xx > -center_mask_w / 2) & (xx < center_mask_w / 2)
    in_center_y = (yy > -center_mask_h / 2) & (yy < center_mask_h / 2)
    mask = ~(in_center_x & in_center_y)
    return np.where(mask, xx, np.nan), np.where(mask, yy, np.nan)

# ============================================================
# ANIMATION UPDATE
# ============================================================
def update(frame):
    cycle_frame = frame % (frames // 2)

    left_radii = []
    right_radii = []

    # 1. วาดวงกลมฝั่งซ้ายและขวา
    for i in range(n_rings):
        r_l = (cycle_frame * wave_speed + i * ring_spacing) % max_radius
        r_r = (cycle_frame * wave_speed + i * ring_spacing) % max_radius
        
        left_radii.append(r_l)
        right_radii.append(r_r)

        # Left Ring
        xx_l, yy_l = get_ring_data(left_source_x, r_l)
        if xx_l is not None:
            xx_l_m, yy_l_m = mask_center(xx_l, yy_l)
            left_rings[i].set_data(xx_l_m, yy_l_m)
            alpha_l = np.clip(r_l / 2.0, 0, 1) * (1 - r_l / max_radius)
            left_rings[i].set_alpha(alpha_l * 0.6)

        # Right Ring
        xx_r, yy_r = get_ring_data(right_source_x, r_r)
        if xx_r is not None:
            xx_r_m, yy_r_m = mask_center(xx_r, yy_r)
            right_rings[i].set_data(xx_r_m, yy_r_m)
            alpha_r = np.clip(r_r / 2.0, 0, 1) * (1 - r_r / max_radius)
            right_rings[i].set_alpha(alpha_r * 0.6)

    # 2. คำนวณจุดตัดแทรกสอด (Interference Points) ตรงกลาง
    ix_list, iy_list = [], []
    d = right_source_x - left_source_x  # ระยะห่างระหว่าง 2 จุดกำเนิด (30)

    for r_l in left_radii:
        for r_r in right_radii:
            # เงื่อนไขการตัดกันของสองวงกลม: |r1 - r2| < d < r1 + r2
            if abs(r_l - r_r) < d < (r_l + r_r):
                # คำนวณพิกัดจุดตัด (Algebraic Circle Intersection)
                a = (r_l**2 - r_r**2 + d**2) / (2 * d)
                h_sq = r_l**2 - a**2
                if h_sq > 0:
                    h = np.sqrt(h_sq)
                    ix = left_source_x + a
                    iy1 = h
                    iy2 = -h

                    # กรองเอาเฉพาะจุดที่อยู่นอกพื้นที่ข้อความตรงกลาง
                    for iy in (iy1, iy2):
                        if not ((-center_mask_w / 2 < ix < center_mask_w / 2) and 
                                (-center_mask_h / 2 < iy < center_mask_h / 2)):
                            if -y_lim < iy < y_lim:
                                ix_list.append(ix)
                                iy_list.append(iy)

    if ix_list:
        interf_dots.set_data(ix_list, iy_list)
        interf_dots.set_alpha(0.7)
        interf_glow.set_data(ix_list, iy_list)
        interf_glow.set_alpha(0.25)
    else:
        interf_dots.set_data([], [])
        interf_glow.set_data([], [])

    return left_rings + right_rings + [interf_dots, interf_glow]

# ============================================================
# RUN ANIMATION
# ============================================================
ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000 / FPS,
    blit=True
)

plt.show()