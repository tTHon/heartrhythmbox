import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. กำหนดสมการเชิงอนุพันธ์ของ Lorenz System
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# 2. คำนวณเส้นทางการเคลื่อนที่ (Integration)
t_span = (0, 50)
t_eval = np.linspace(0, 50, 10000) # ความละเอียดของเส้น
initial_state = [1.0, 1.0, 1.0]   # จุดเริ่มต้น

solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
x, y, z = solution.y

# 3. ตั้งค่าการวาดกราฟ 3 มิติ
fig = plt.figure(figsize=(16, 12), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# 4. วาดเส้นสีแดงแบบโปร่งแสงเพื่อให้เห็นมิติการซ้อนทับ
ax.plot(x, y, z, color='#ff2a2a', lw=0.6, alpha=0.55)

# 5. เพิ่มจุดวงกลมสีเขียว (แทนตำแหน่งสถานะ ณ เวลาหนึ่ง)
point_idx = 1850
#ax.scatter(x[point_idx], y[point_idx], z[point_idx], 
#           color='#39ff14', s=180, edgecolors='darkgreen', zorder=5)

# 6. ปรับมุมมองและซ่อนแกนกราฟให้เหมือนในภาพตัวอย่าง
ax.view_init(elev=15, azim=120) # ปรับมุมกล้อง (ก้ม/เงย, หมุนซ้าย/ขวา)
ax.set_axis_off()               # ปิดเส้นแกน X, Y, Z

plt.tight_layout()
plt.show()
