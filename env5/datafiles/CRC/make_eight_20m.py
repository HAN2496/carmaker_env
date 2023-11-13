import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

cone_r = 0.2
car_width, car_length = 1.8, 4

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

straight_line = np.array([[x, 6.26741445879301E-06] for x in np.linspace(0, 100, int((100 - 0) / 0.01) + 1)])
straight_line2 = np.array([[x, 6.26741445879301E-06] for x in np.linspace(100, 200, int((200 - 100) / 0.01) + 1)])
x0, y0 = 100, 6.26741445879301E-06 + 20
r = 20

circle_path = create_circle_path(x0, y0, r, 0)
circle_path = np.array(circle_path)

circle_path2 = create_circle_path(x0, y0 - 40, r, 0)
circle_path2 = np.array(circle_path2)

total_traj = np.vstack((straight_line, circle_path, circle_path2, straight_line2))

plt.figure(figsize=(6,6))
plt.plot(straight_line[:, 0], straight_line[:, 1], marker='.')
plt.plot(circle_path[:, 0], circle_path[:, 1], marker='.')
plt.plot(circle_path2[:, 0], circle_path2[:, 1], marker='.')
plt.plot(straight_line2[:, 0], straight_line2[:, 1], marker='.')
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
