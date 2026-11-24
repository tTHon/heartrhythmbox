import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 1. คำนวณสมการ Lorenz Attractor
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

t_eval = np.linspace(0, 75, 30000)
sol = solve_ivp(lorenz, (0, 75), [1.0, 1.0, 1.0], t_eval=t_eval)
x, y, z = sol.y

# 2. ตั้งค่า Figure พื้นหลังสีดำสนิท
fig = plt.figure(figsize=(10, 10), facecolor='#050505')

# ==========================================
# ส่วนที่ 1: เลเยอร์พื้นหลังวงกลมเรดาร์ (Polar Rings)
# ==========================================
ax_polar = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, facecolor='#050505')

ax_polar.set_theta_zero_location('N')
ax_polar.set_theta_direction(-1)

# วาดวงกลมซ้อนกัน 5 ชั้น (รัศมี 2, 4, 6, 8, 10)
radii = np.arange(2, 11, 2)
ax_polar.set_ylim(0, 10.5)
ax_polar.set_yticks(radii)
ax_polar.set_yticklabels([])

# วาดเส้นรัศมีแบ่งองศาทุกๆ 30 องศา
angles_deg = np.arange(0, 360, 30)
ax_polar.set_xticks(np.deg2rad(angles_deg))
ax_polar.set_xticklabels([f"{deg}°" for deg in angles_deg], 
                         color='#4a5568', fontsize=8, family='monospace')
ax_polar.tick_params(axis='x', pad=8)

ax_polar.grid(color='#222730', linestyle='-', linewidth=0.8)
ax_polar.spines['polar'].set_color('#3a414d')
ax_polar.spines['polar'].set_linewidth(1.0)

# เพิ่มจุดไข่ปลาเล็กๆ (Dot Ring) ที่รัศมี 3 และ 7 (แก้จุด SyntaxError บรรทัด 48)
dot_angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
for r in (3, 7):
    ax_polar.scatter(dot_angles, np.full_like(dot_angles, r), 
                     color='#2d3748', s=2.5, alpha=0.7)

# ==========================================
# ส่วนที่ 2: วาดกราฟ Lorenz Attractor 3 มิติซ้อนทับตรงกลาง
# ==========================================
ax_3d = fig.add_axes([0.12, 0.12, 0.76, 0.76], projection='3d')
ax_3d.patch.set_alpha(0.0)

colors = ["#d91e2a", "#ff5e3a", "#fff2cc", "#4deeea", "#3b4cca"]
custom_cmap = LinearSegmentedColormap.from_list("crosstalk_glow", colors, N=256)

points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(-20, 20)

# ซ้อนเส้น 3 ชั้นสร้างเอฟเฟกต์เรืองแสง (Glow)
for lw, alpha in ((6.5, 0.06), (2.5, 0.15), (0.75, 0.65)):
    lc = Line3DCollection(segments, cmap=custom_cmap, norm=norm, alpha=alpha, linewidth=lw)
    lc.set_array(x)
    ax_3d.add_collection(lc)

ax_3d.set_xlim(x.min(), x.max())
ax_3d.set_ylim(y.min(), y.max())
ax_3d.set_zlim(z.min(), z.max())
ax_3d.view_init(elev=10, azim=90)
ax_3d.set_axis_off()

#plt.savefig('lorenz_polar_radar.png', dpi=300, bbox_inches='tight', facecolor='#050505')
plt.show()
