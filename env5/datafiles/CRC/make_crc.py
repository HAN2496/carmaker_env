import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd

cone_r = 0.2
car_width, car_length = 1.8, 4

def create_enter_path(x):
    return 1.16154577 * 10e-4 * x ** 3 - 1.91708131 * 10e-2 * x ** 2 + 0.35866151 * x + 34.1823615

def create_circle_path(x0, y0, r, direction, step=0.01):
    if direction == 0:
        angles = np.arange(np.pi * 1.5, np.pi * 1.5 + 2 * np.pi, step)
    else:
        angles = np.arange(np.pi * 1.5, np.pi * 1.5 - 2 * np.pi, -step)

    path = []

    for angle in angles:
        x = x0 + r * np.cos(angle)
        y = y0 + r * np.sin(angle)
        path.append((x, y))

    return path


straight_path = np.array([[x, -5.5] for x in np.linspace(0, 100, int((100 - 0) / 0.01) + 1)])

tmpx = 130 - 30 * np.sin(13 / 6 - np.pi / 2)
tmpy = 30 + 30 * np.cos(13 / 6 - np.pi / 2)
x = np.array([100, tmpx])
y = np.array([-5.5, tmpy])
dydx = np.array([0, - tmpx / tmpy])
cs = CubicSpline(x, y, bc_type=((1, dydx[0]), (1, dydx[1])))

#enter_path = np.array([[x, cs(x)] for x in np.linspace(100, tmpx, int((tmpx - 100) / 0.01) + 1)])

enter_path = np.array([[x, create_enter_path(x)] for x in np.linspace(100, 130, int((130 - 100) / 0.01) + 1)])

x0, y0 = 100, 30
r = 30
circle_path = create_circle_path(x0, y0, r, 0)
circle_path = np.array(circle_path)


total_traj = np.vstack((straight_path, circle_path))

plt.figure(figsize=(6,6))
plt.plot(straight_path[:, 0], straight_path[:, 1], marker='.')
plt.plot(enter_path[:, 0], enter_path[:, 1], marker='.')
plt.plot(circle_path[:, 0], circle_path[:, 1], marker='.')
#plt.plot(total_traj[:, 0], total_traj[:, 1], marker='.')
plt.scatter(x0, y0, color='red') # Mark the center
plt.axis('equal') # Equal scaling for both axes
plt.grid(True)
plt.show()


df = pd.DataFrame({
    'traj_tx': total_traj[:, 0],
    'traj_ty': total_traj[:, 1]
})

df.to_csv("traj_eight.csv", index=False)
