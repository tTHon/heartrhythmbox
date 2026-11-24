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
# ส่วนที่ 1: สร้างเลเยอร์พื้นหลัง (กรอบ + เส้นกริด 2D)
# ==========================================
ax_bg = fig.add_axes([0.1, 0.1, 0.8, 0.8], facecolor='#050505')

# วาดเส้นกริดหลักและกริดย่อย
ax_bg.set_xlim(0, 10)
ax_bg.set_ylim(0, 10)
ax_bg.set_xticks(np.arange(0, 11, 2))
ax_bg.set_yticks(np.arange(0, 11, 2))
ax_bg.set_xticks(np.arange(0, 10.5, 0.5), minor=True)
ax_bg.set_yticks(np.arange(0, 10.5, 0.5), minor=True)

ax_bg.grid(which='major', color='#222730', linestyle='-', linewidth=0.9)
ax_bg.grid(which='minor', color='#12151a', linestyle='-', linewidth=0.5)

# ตกแต่งกรอบสี่เหลี่ยมรอบนอก
for spine in ax_bg.spines.values():
    spine.set_edgecolor('#3a414d')
    spine.set_linewidth(1.2)
ax_bg.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ==========================================
# ส่วนที่ 2: วาดกราฟ Lorenz Attractor 3 มิติซ้อนทับ
# ==========================================
ax_3d = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='3d')
ax_3d.patch.set_alpha(0.0) # ทำให้พื้นหลังโปร่งใสเพื่อเห็นกริดด้านหลัง

colors = ["#d91e2a", "#ff5e3a", "#fff2cc", "#4deeea", "#3b4cca"]
custom_cmap = LinearSegmentedColormap.from_list("crosstalk_glow", colors, N=256)

points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(-20, 20)

# ซ้อนเส้น 3 ชั้นสร้างแสงฟุ้ง (Glow)
for lw, alpha in [(6.5, 0.06), (2.5, 0.15), (0.75, 0.65)]:
    lc = Line3DCollection(segments, cmap=custom_cmap, norm=norm, alpha=alpha, linewidth=lw)
    lc.set_array(x)
    ax_3d.add_collection(lc)

ax_3d.set_xlim(x.min(), x.max())
ax_3d.set_ylim(y.min(), y.max())
ax_3d.set_zlim(z.min(), z.max())
ax_3d.view_init(elev=10, azim=90)
ax_3d.set_axis_off()

# ==========================================
# ส่วนที่ 3: เพิ่มจุดพิกัดและตัวเลขชี้บอกค่า (Data Callouts)
# ==========================================
ax_hud = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax_hud.patch.set_alpha(0.0)
ax_hud.set_xlim(0, 10)
ax_hud.set_ylim(0, 10)
ax_hud.set_axis_off()

# ฟังก์ชันช่วยวาดจุดและเส้นชี้ข้อความ
def add_callout(ax, dot_xy, text_xy, label):
    ax.scatter(*dot_xy, color='#8fa0b5', s=12, zorder=10)
    ax.plot([dot_xy[0], text_xy[0]], [dot_xy[1], text_xy[1]], color='#4a5568', lw=0.7)
    ax.text(text_xy[0] + 0.1, text_xy[1], label, color='#8fa0b5',
            fontsize=8, family='monospace', va='center')

# วางจุดอ้างอิงตามตำแหน่งต่างๆ (ปรับแก้ตัวเลขและพิกัดได้ตามต้องการ)
#add_callout(ax_hud, (4.2, 8.2), (4.6, 8.2), "σ = 10.000")
#add_callout(ax_hud, (4.2, 7.9), (4.6, 7.9), "ρ = 28.000")
#add_callout(ax_hud, (1.8, 2.8), (2.2, 2.8), "Δt = 0.0025")
#add_callout(ax_hud, (1.8, 2.5), (2.2, 2.5), "β = 2.667")
#add_callout(ax_hud, (8.1, 3.2), (7.2, 3.2), "X-TALK SYNC")

# บันทึกภาพความละเอียดสูง
#plt.savefig('lorenz_hud_poster.png', dpi=300, bbox_inches='tight', facecolor='#050505')
plt.show()
