import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

cone_r = 0.2
car_width, car_length = 1.568, 4.3
xcoords = []
ycoords = []

cones = np.array([(100 + 30 * i, -5.25) for i in range(10)])

straight_line1 = np.array([[x, -5.25] for x in np.arange(0, 85, 0.1)])
straight_line2 = np.array([[x, -5.25] for x in np.arange(385, 500, 0.1)])

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

interpolate_first = cubic_interpolation(100 - 15, -5.25, 100, -5.25 + cone_r)
interpolate_last = cubic_interpolation(370, -5.25 - cone_r, 370 + 15)

plt.scatter([100 + 30 * i for i in range(10)], [-5.25 for i in range(10)], label='cones')
plt.plot(xcoords, ycoords, '-', label='Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
#plt.axis('equal')
plt.show()

data = np.array([xcoords, ycoords]).T
data_sorted = data[data[:, 0].argsort()]

xcoords_sorted = data_sorted[:, 0]
ycoords_sorted = data_sorted[:, 1]

df = pd.DataFrame({
    'traj_tx': xcoords_sorted,
    'traj_ty': ycoords_sorted
})

df.to_csv("traj_slalom_onefifth.csv", index=False)
