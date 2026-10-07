import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 1. Grid ความละเอียดสูง (Top View)
N = 1000
x = np.linspace(-2.2, 2.2, N)
y = np.linspace(-2.2, 2.2, N)
X, Y = np.meshgrid(x, y)

# 2. พิกัดหมุนแกนรูปทรง
angle_z = np.radians(65)
X_rot = X * np.cos(angle_z) - Y * np.sin(angle_z)
Y_rot = X * np.sin(angle_z) + Y * np.cos(angle_z)

R = np.sqrt((X_rot / 1.30)**2 + (Y_rot / 0.85)**2)
THETA = np.arctan2(Y_rot, X_rot)

# 3. สนามความสูงรูปทรงเกลียวดั้งเดิม
spiral_phase = 2 * THETA - 3.2 * np.log(R + 0.12)
Z_base = np.exp(-1.15 * R) * (1 + 0.28 * np.sin(spiral_phase))

tilt_angle_y = np.radians(55) * np.exp(-0.8 * R) + np.radians(15)
Z_tilted = Z_base + 0.25 * X * np.sin(tilt_angle_y)

# 4. แบ่ง Shell 9 ชั้น
num_shells = 9
levels = np.linspace(Z_tilted.min(), Z_tilted.max() + 1e-5, num_shells + 1)

# เฉดสีแก้วโปร่งแสงตามรูปต้นแบบ (RGBA)
colors_rgba = np.array([
    [0.02, 0.08, 0.16, 0.40], # 1. ครามมืด
    [0.05, 0.22, 0.35, 0.42], # 2. ฟ้าเข้ม
    [0.08, 0.38, 0.48, 0.45], # 3. ฟ้ามรกต
    [0.13, 0.55, 0.52, 0.48], # 4. เขียวฟ้า
    [0.25, 0.72, 0.48, 0.50], # 5. เขียวสว่าง
    [0.52, 0.76, 0.31, 0.52], # 6. เขียวตอง
    [0.89, 0.58, 0.11, 0.55], # 7. ส้มทอง
    [0.86, 0.30, 0.23, 0.58], # 8. ส้มแดง
    [0.98, 0.88, 0.85, 0.65]  # 9. ขาวใจกลาง
])

# 5. สังเคราะห์ภาพแบบ Alpha Blending ซ้อนแผ่นแก้วโปร่งแสง
canvas = np.zeros((N, N, 3), dtype=float)

for i in range(num_shells):
    mask = Z_tilted >= levels[i]
    r, g, b, alpha = colors_rgba[i]
    
    # ผสมสีให้แผ่นชั้นล่างมองเห็นส่องทะลุขึ้นมา
    for c, val in enumerate([r, g, b]):
        canvas[:, :, c] = np.where(mask, canvas[:, :, c] * (1 - alpha) + val * alpha, canvas[:, :, c])

fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_facecolor('black')

# 6. วาดเนื้อแผ่นแก้วโปร่งใส
ax.imshow(canvas, origin='lower', extent=[-2.2, 2.2, -2.2, 2.2])

# 7. *** สร้างมิติความหนาสันแก้ว (Beveled Rim & Drop Shadow Offset) ***
for i in range(1, num_shells + 1):
    lvl = levels[i]
    
    # 7.1 เงามืดใต้แผ่นแก้ว (Drop Shadow Offset) ขยับไปทางขวาล่างเล็กน้อย
    ax.contour(X - 0.012, Y - 0.012, Z_tilted, levels=[lvl], colors='#000000', linewidths=3.0, alpha=0.70)
    
    # 7.2 เส้นขอบแก้วตัดคม
    ax.contour(X, Y, Z_tilted, levels=[lvl], colors='#06101c', linewidths=1.0, alpha=0.85)
    
    # 7.3 สันขอบแก้วสะท้อนแสง (Glossy Rim Highlight) ขยับไปทางซ้ายบน
    ax.contour(X + 0.010, Y + 0.010, Z_tilted, levels=[lvl], colors='#ffffff', linewidths=1.2, alpha=0.55)

# 8. เส้นวิถีสีขาว และจุดสว่างตรงกลาง
t = np.linspace(-2.5, 2.5, 1000)

x1 = -0.32 * t + 0.04 * np.tanh(3.0 * t)
y1 =  1.00 * t + 0.06 * np.exp(-t**2)

x2 = -0.26 * t - 0.04 * np.tanh(3.0 * t)
y2 =  1.00 * t - 0.06 * np.exp(-t**2)

# Glow เส้นขาว
ax.plot(x1, y1, color='#ffeaa7', linewidth=6.0, alpha=0.20)
ax.plot(x2, y2, color='#ffeaa7', linewidth=6.0, alpha=0.20)

# เส้นวิถี
ax.plot(x1, y1, color='white', linewidth=1.6, alpha=0.98)
ax.plot(x2, y2, color='white', linewidth=1.8, alpha=0.98)

# จุดปะทะเรืองแสง
ax.scatter([0], [0], color='#ffffff', s=350, alpha=0.4, zorder=10)
ax.scatter([0], [0], color='#ffeaa7', s=150, alpha=0.7, zorder=11)
ax.scatter([0], [0], color='#ffffff', s=50,  alpha=1.0, zorder=12)

# 9. ตั้งค่าการแสดงผล
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect('equal')
plt.axis('off')

plt.tight_layout()
plt.show()