import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

cone_r = 0.2
car_width, car_length = 1.8, 4

def make_dlc_cone(start, interval, width, num):
    arr = []
    for i in range(num):
        arr.append([start[0] + i * interval, start[1] + width / 2])
        arr.append([start[0] + i * interval, start[1] - width / 2])
    return arr

cone1 = make_dlc_cone([50, -10], 3, 2.23, 5)
cone2 = make_dlc_cone([75.5, -6.485], 2.75, 2.8, 5)
cone3 = make_dlc_cone([99, -10.385], 3, 3, 5)
cones = np.array(cone1 + cone2 + cone3)

straight_line1 = np.array([[x, -10] for x in np.linspace(0, 62, int((62 - 0) / 0.01) + 1)])
straight_line2 = np.array([[x, -6.485] for x in np.linspace(75.5, 86.5, int((86.5 - 75.5) / 0.01) + 1)])
straight_line3 = np.array([[x, -10.385] for x in np.linspace(99, 200, int((200 - 99) / 0.01) + 1)])


def cubic_interpolation(x0, y0, x1, y1):
    a = 2 * (y0 - y1) / (x1 - x0) ** 3
    b = -3 / 2 * a * (x1 - x0)
    d = y0

    def interpolator(x):
        return a * (x - x0) ** 3 + b * (x - x0) ** 2 + d

    f = interpolator
    x = np.linspace(x0, x1, int((x1 - x0) / 0.01)+1)
    y = [f(xi) for xi in x]

    return np.array(list(zip(x, y)))

interpolate1 = cubic_interpolation(62, -10, 75.5, -6.485)
interpolate2 = cubic_interpolation(86.5, -6.485, 99, -10.385)

total_traj = np.vstack((straight_line1, interpolate1, straight_line2, interpolate2, straight_line3))

plt.scatter(cones[:, 0], cones[:, 1], label='cone', color='r')
plt.plot(total_traj[:, 0], total_traj[:, 1], label='Trajectory', color='orange')
plt.legend()
plt.grid(True)
#plt.axis('equal')
plt.show()


df = pd.DataFrame({
    'traj_tx': total_traj[:, 0],
    'traj_ty': total_traj[:, 1]
})

df.to_csv("traj_dlc.csv", index=False)
