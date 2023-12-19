import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

CONER = 0.2
CARWIDTH = 1.975
CARLENGTH = 4.94
dist_from_axis = CARWIDTH / 2 + CONER

df = pd.read_csv(f"total_traj.csv").loc[:, ["traj_tx", "traj_ty"]].values
lines = []
def cubic_interpolation(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    a = 2 * (y0 - y1) / (x1 - x0) ** 3
    b = -3 / 2 * a * (x1 - x0)
    d = y0

    def interpolator(x):
        return a * (x - x0) ** 3 + b * (x - x0) ** 2 + d

    f = interpolator
    x = np.linspace(x0, x1, int((x1 - x0) / 0.01)+1)
    y = [f(xi) for xi in x]

    return np.array(list(zip(x, y)))

"""
section1
Lane Change
"""
before_lanechange = np.array([[x, -12.25] for x in np.linspace(0, 150, int((150 - 0) / 0.01) + 1)])
interpolate1 = cubic_interpolation([150, -12.25], [175, -9.63875])
straight = np.array([[x, -9.63875] for x in np.linspace(175, 186.5, int((186.5-175) / 0.01) + 1)])
interpolate2 = cubic_interpolation([186.5, -9.63875], [199, -12.53875])

section1 = np.vstack((before_lanechange, interpolate1, straight, interpolate2))

"""
section3
SLALOM
"""
before_slalom = cubic_interpolation([199, -12.53875], [230, -12.25])
before_slalom2 = np.array([[x, -12.25] for x in np.linspace(230, 400-15, int((385-230) / 0.01) + 1)])

slalom_start = 350
interpolated = cubic_interpolation([slalom_start + 35, -12.25], [slalom_start + 50, -12.25 + dist_from_axis])

section2 = np.vstack((before_slalom, before_slalom2, interpolated))


for i in range(9):
    sign = 0
    if i % 2 == 0:
        sign = 1
    else:
        sign = -1
    interpolated = cubic_interpolation([slalom_start + 50 + 30 * i, -12.25 + dist_from_axis * sign],
                                       [slalom_start + 50 + 30 * (i + 1), -12.25 - dist_from_axis * sign])
    section2 = np.vstack((section2, interpolated))

interpolated = cubic_interpolation([slalom_start + 320, -12.25 - dist_from_axis], [slalom_start + 320 + 15, -12.25])
straight_line2 = np.array([[x, -12.25] for x in np.arange(slalom_start + 320 + 15, slalom_start + 400, 0.01)])

section2 = np.vstack((section2, straight_line2, interpolated))


others = df[:5575, :]
print(section2)
total_traj = np.vstack((section2, others))
df = pd.DataFrame({
    'traj_tx': total_traj[:, 0],
    'traj_ty': total_traj[:, 1]
})

df.to_csv("traj_dlc.csv", index=False)

"""
lines.append(section2)

section 03. Ramp
"""
"""
slalom_end = slalom_start + 400
lines = []
for i in range(27):
    lines.append(df[5575 + i * 1000: 5575 + (i + 1) * 1000, :])

lines = np.array(lines)
print(lines[-1])

total = np.vstack((section1, section2))
plt.scatter(total[:, 0], total[:, 1], label='Trajectory', color='orange')
plt.legend()
plt.grid(True)
#plt.axis('equal')
#plt.show()
"""