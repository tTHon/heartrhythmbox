import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 1. สมการ Lorenz Attractor
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# 2. คำนวณเส้นทาง (ใช้เวลาให้นานขึ้นเพื่อให้เส้นซ้อนกันเป็นริ้วสวยงาม)
t_eval = np.linspace(0, 75, 30000)
sol = solve_ivp(lorenz, (0, 75), [1.0, 1.0, 1.0], t_eval=t_eval)
x, y, z = sol.y

# 3. สร้าง Custom Colormap ให้ตรงกับสีในรูปตัวอย่าง
# ไล่สีจากซ้าย (ค่า X ติดลบ) -> กลาง (X = 0) -> ขวา (ค่า X เป็นบวก)
colors = [
    "#d91e2a",  # แดงเข้ม (ขอบซ้าย)
    "#ff5e3a",  # ส้มอมแดง
    "#fff2cc",  # ขาวอมเหลืองสว่าง (จุดตัดตรงกลาง)
    "#4deeea",  # ฟ้าสว่าง (ไซอัน)
    "#3b4cca"   # น้ำเงินอมม่วง (ขอบขวา)
]
custom_cmap = LinearSegmentedColormap.from_list("crosstalk_glow", colors, N=256)

# เตรียมข้อมูลเส้นสำหรับทำ Gradient
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 4. ตั้งค่ากราฟพื้นหลังสีดำสนิท
fig = plt.figure(figsize=(16, 8), facecolor='none')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('none')

# กำหนดค่าสีตามแกน X (ปรับช่วง -20 ถึง 20 เพื่อให้สีขาวอยู่ตรงกลางจุดตัดพอดี)
norm = plt.Normalize(-20, 20)

# 5. วาดเส้นซ้อน 3 ชั้นเพื่อสร้างเอฟเฟกต์แสงฟุ้ง (Glow / Bloom)
# ชั้นที่ 1: แสงฟุ้งวงกว้าง (เส้นหนา จางมาก)
lc_glow_wide = Line3DCollection(segments, cmap=custom_cmap, norm=norm, alpha=0.06, linewidth=6.0)
lc_glow_wide.set_array(x)
ax.add_collection(lc_glow_wide)

# ชั้นที่ 2: แสงฟุ้งชั้นกลาง
lc_glow_mid = Line3DCollection(segments, cmap=custom_cmap, norm=norm, alpha=0.15, linewidth=2.5)
lc_glow_mid.set_array(x)
ax.add_collection(lc_glow_mid)

# ชั้นที่ 3: เส้นแกนกลางคมชัด
lc_core = Line3DCollection(segments, cmap=custom_cmap, norm=norm, alpha=0.65, linewidth=0.7)
lc_core.set_array(x)
ax.add_collection(lc_core)

# 6. จัดมุมกล้องให้เห็นปีก 2 ข้างสมมาตรเหมือนในโปสเตอร์
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())
ax.view_init(elev=10, azim=90) # มุมมองด้านหน้าตรงเพื่อเห็นปีกซ้าย-ขวาชัดเจน
ax.set_axis_off()

plt.tight_layout()
# บันทึกภาพความละเอียดสูง (พื้นหลังโปร่งใสหรือสีดำ)
plt.savefig('playground/xWaves/lorenz_glow.png', dpi=300, bbox_inches='tight', facecolor='none')
#plt.show()
