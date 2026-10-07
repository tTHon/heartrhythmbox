import numpy as np
import matplotlib.pyplot as plt

# 1. สร้าง Grid
N = 600
x = np.linspace(-2.2, 2.2, N)
y = np.linspace(-2.2, 2.2, N)
X, Y = np.meshgrid(x, y)

# 2. คำนวณทรงเกลียว
angle_z = np.radians(65)
X_rot = X * np.cos(angle_z) - Y * np.sin(angle_z)
Y_rot = X * np.sin(angle_z) + Y * np.cos(angle_z)
R = np.sqrt((X_rot / 1.30)**2 + (Y_rot / 0.85)**2)
THETA = np.arctan2(Y_rot, X_rot)

spiral_phase = 2 * THETA - 3.2 * np.log(R + 0.12)
Z_base = np.exp(-1.15 * R) * (1 + 0.28 * np.sin(spiral_phase))
Z_tilted = Z_base + 0.25 * X * np.sin(np.radians(55) * np.exp(-0.8 * R) + np.radians(15))

# 3. สร้าง Contour Lines เพื่อส่งออก SVG
fig, ax = plt.subplots(figsize=(10, 10))
num_shells = 9
levels = np.linspace(Z_tilted.min(), Z_tilted.max(), num_shells + 1)

# เซฟเฉพาะเส้นเป็น SVG เวกเตอร์
cs = ax.contour(X, Y, Z_tilted, levels=levels, colors='black')
plt.axis('off')
plt.savefig("playground/xWaves/shells_vector.svg", format="svg", bbox_inches='tight', pad_inches=0)
print("ส่งออกไฟล์ shells_vector.svg เรียบร้อยแล้ว!")