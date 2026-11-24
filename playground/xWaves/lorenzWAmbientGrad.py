import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 1. คำนวณสมการ Lorenz Attractor
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)

t_eval = np.linspace(0, 95, 30000)
sol = solve_ivp(lorenz, (0, 95), (1.0, 1.0, 1.0), t_eval=t_eval)
x, y, z = sol.y

# 2. ตั้งค่า Figure สัดส่วน 16:9 (16 x 9 นิ้ว)
fig = plt.figure(figsize=(51, 12), facecolor='#050505')

# ==========================================
# ส่วนที่ 1: พื้นหลังแสงฟุ้ง 2 สี (ปรับสัดส่วนให้รับกับจอ 16:9)
# ==========================================
ax_bg = fig.add_axes((0, 0, 1, 1), facecolor='#050505')

res = 800
gx = np.linspace(-1.77, 1.77, int(res * 1.77))
gy = np.linspace(-1, 1, res)
X, Y = np.meshgrid(gx, gy)

# แสงฟุ้งฝั่งซ้าย (แดง) และฝั่งขวา (น้ำเงิน) กระจายตัวกว้างรับจอ 16:9
glow_left = np.exp(-((X + 0.85)**2 / 0.2 + (Y - 0.0)**2 / 0.2))
glow_right = np.exp(-((X - 0.85)**2 / 0.2 + (Y - 0.0)**2 / 0.2))

rows, cols = X.shape
img = np.full((rows, cols, 3), 0.02)

# เติมสีแดงฝั่งซ้าย
img[:, :, 0] += glow_left * 0.24
img[:, :, 1] += glow_left * 0.035
img[:, :, 2] += glow_left * 0.035

# เติมสีน้ำเงินฝั่งขวา
img[:, :, 0] += glow_right * 0.03
img[:, :, 1] += glow_right * 0.14
img[:, :, 2] += glow_right * 0.36

ax_bg.imshow(np.clip(img, 0, 1), extent=(-1.77, 1.77, -1, 1), origin='lower', aspect='auto')
ax_bg.set_axis_off()

# ==========================================
# ส่วนที่ 2: วาดกราฟ Lorenz Attractor 3 มิติ (ขยายใหญ่เต็มพื้นที่ 16:9)
# ==========================================
# ขยายกรอบ 3D ให้ล้นขอบบน-ล่าง (-0.18 ถึง 1.18) เพื่อตัดขอบโปร่งใสรอบกล่อง 3D ทิ้ง
ax_3d = fig.add_axes((-0.05, -0.18, 1.10, 1.36), projection='3d')
ax_3d.patch.set_alpha(0.0)

colors = ("#3b4cca", "#4deeea", "#fff2cc", "#ff5e3a", "#d91e2a")
custom_cmap = LinearSegmentedColormap.from_list("crosstalk_glow", colors, N=256)

points = np.array((x, y, z)).T.reshape(-1, 1, 3)
segments = np.concatenate((points[:-1], points[1:]), axis=1)
norm = plt.Normalize(-20, 20)

for lw, alpha in ((7.0, 0.06), (2.8, 0.15), (0.85, 0.65)):
    lc = Line3DCollection(segments, cmap=custom_cmap, norm=norm, alpha=alpha, linewidth=lw)
    lc.set_array(x)
    ax_3d.add_collection(lc)

# ตั้งขอบเขตแกนให้พอดีกับข้อมูลจริง ไม่ให้ปีกขวาโดนตัด
ax_3d.set_xlim(x.min(), x.max())
ax_3d.set_ylim(y.min(), y.max())
ax_3d.set_zlim(z.min(), z.max())

# ปรับสัดส่วนกล่อง 3D ให้ผายออกแนวนอนรับกับจอ 16:9 และซูมกล้องเข้าใกล้
ax_3d.set_box_aspect((1.35, 1.35, 1.0), zoom=1.2)
ax_3d.view_init(elev=12, azim=111)
ax_3d.set_axis_off()

# บันทึกภาพความละเอียดสูง (16:9)
plt.savefig('playground/xWaves/lorenz.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='#050505')
#plt.show()
