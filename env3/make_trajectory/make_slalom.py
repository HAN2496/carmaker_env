import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

cone_r = 0.2
car_width, car_length = 1.568, 4.
dist_from_axis = (car_width + 1) / 2 + cone_r

cones = np.array([(100 + 30 * i, -5.25) for i in range(10)])

def cubic_interpolation(x0, y0, x1, y1, interval=0.01):
    a = 2 * (y0 - y1) / (x1 - x0) ** 3
    b = -3 / 2 * a * (x1 - x0)
    d = y0

    def interpolator(x):
        return a * (x - x0) ** 3 + b * (x - x0) ** 2 + d

    f = interpolator
    x = np.linspace(x0, x1, int((x1 - x0) / interval)+1)
    y = [f(xi) for xi in x]

    return np.array(list(zip(x, y)))

straight_line1 = np.array([[x, -5.25] for x in np.arange(0, 85, 0.01)])
straight_line2 = np.array([[x, -5.25] for x in np.arange(385, 600, 0.01)])

interpolate_first = cubic_interpolation(100 - 15, -5.25, 100, -5.25 + dist_from_axis)
interpolate_last = cubic_interpolation(370, -5.25 - dist_from_axis, 370 + 15, -5.25)


total_traj = np.vstack((straight_line1, straight_line2, interpolate_first, interpolate_last))

for i in range(9):
    dist_from_axis = (car_width + 1) / 2 + cone_r
    sign = 0
    if i % 2 == 0:
        sign = 1
    else:
        sign = -1
    interpolated = cubic_interpolation(100 + 30 * i, -5.25 + dist_from_axis * sign, 100 + 30 * (i + 1), -5.25 - dist_from_axis * sign)
    total_traj = np.vstack((total_traj, interpolated))

data = np.array([total_traj[:, 0], total_traj[:, 1]]).T
data_sorted = data[data[:, 0].argsort()]

plt.scatter([100 + 30 * i for i in range(10)], [-5.25 for i in range(10)], label='cones')
plt.plot(data_sorted[:, 0], data_sorted[:, 1], '-', label='Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
#plt.axis('equal')
plt.show()

xcoords_sorted = data_sorted[:, 0]
ycoords_sorted = data_sorted[:, 1]

df = pd.DataFrame({
    'traj_tx': xcoords_sorted,
    'traj_ty': ycoords_sorted
})

df.to_csv("traj_slalom.csv", index=False)


dist_from_axis_at_straight = 1.1 * car_width + 0.25
dist_from_axis_at_straight = dist_from_axis_at_straight / 2

cone_straight_line1_upper = np.array([[x, -5.25 + dist_from_axis_at_straight] for x in np.arange(0, 85, 5)])
cone_straight_line1_lower = np.array([[x, -5.25 - dist_from_axis_at_straight] for x in np.arange(0, 85, 5)])
cone_straight_line2_upper = np.array([[x, -5.25 + dist_from_axis_at_straight] for x in np.arange(385, 600, 5)])
cone_straight_line2_lower = np.array([[x, -5.25 - dist_from_axis_at_straight] for x in np.arange(385, 600, 5)])

total_cone = np.vstack((cone_straight_line1_upper, cone_straight_line2_upper, cone_straight_line1_lower, cone_straight_line2_lower))

cone_interpolate_first_upper = cubic_interpolation(100 - 15, -5.25 + dist_from_axis_at_straight, 100, -5.25 + dist_from_axis + cone_r, interval=0.001)
cone_interpolate_last_upper = cubic_interpolation(370, -5.25 - cone_r, 370 + 15, -5.25 + dist_from_axis_at_straight, interval=0.001)
cone_interpolate_first_lower = cubic_interpolation(100 - 15, -5.25 - dist_from_axis_at_straight, 100, -5.25 + cone_r, interval=0.001)
cone_interpolate_last_lower = cubic_interpolation(370, -5.25 - dist_from_axis - cone_r, 370 + 15, -5.25 - dist_from_axis_at_straight, interval=0.001)

total_cone = np.vstack((total_cone, cone_interpolate_first_upper, cone_interpolate_last_upper, cone_interpolate_first_lower, cone_interpolate_last_lower))

for i in range(9):
    dist_from_axis = (car_width + 1) / 2 + cone_r
    if i % 2 == 0:
        interpolated_upper = cubic_interpolation(100 + 30 * i, -5.25 + (dist_from_axis + cone_r), 100 + 30 * (i + 1), -5.25 - cone_r, interval=0.5)
        interpolated_lower = cubic_interpolation(100 + 30 * i, -5.25 + cone_r, 100 + 30 * (i + 1), -5.25 - (dist_from_axis + cone_r), interval=0.5)
    else:
        interpolated_upper = cubic_interpolation(100 + 30 * i, -5.25 - cone_r, 100 + 30 * (i + 1), -5.25 + (dist_from_axis + cone_r), interval=0.5)
        interpolated_lower = cubic_interpolation(100 + 30 * i, -5.25 - (dist_from_axis + cone_r), 100 + 30 * (i + 1), -5.25 + cone_r, interval=0.5)

    total_cone = np.concatenate((total_cone, interpolated_upper, interpolated_lower), axis=0)

cone = np.array([total_cone[:, 0], total_cone[:, 1]]).T
cone_sorted = total_cone[total_cone[:, 0].argsort()]

plt.scatter([100 + 30 * i for i in range(10)], [-5.25 for i in range(10)], label='cones')
plt.scatter(cone_sorted[:, 0], cone_sorted[:, 1], label='Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
#plt.axis('equal')
plt.show()
