import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

cone_r = 0.2
car_width, car_length = 1.568, 4.3
xcoords = []
ycoords = []

cones = np.array([(100 + 30 * i, -5.25) for i in range(10)])

straight_line1 = np.array([[x, -5.25, 0] for x in np.arange(0, 85, 0.1)])
straight_line2 = np.array([[x, -5.25, 0] for x in np.arange(385, 500, 0.1)])
straight_line1x = np.linspace(0, 85, int((85 - 0) / 0.01) + 1)
xcoords.extend(straight_line1x)
straight_line1y = np.full_like(straight_line1x, -5.25)
ycoords.extend(straight_line1y)

straight_line2x = np.linspace(385, 500, int((500 - 385) / 0.01) + 1)
xcoords.extend(straight_line2x)
straight_line2y = np.full_like(straight_line2x, -5.25)
ycoords.extend(straight_line2y)

for idx, (conex, coney) in enumerate(cones):
    height = cone_r + car_width / 2
    if idx == 0:
        x1, x2 = conex - 15, conex
        y1, y2 = -5.25, -5.25 + height
        dy = [0, 0]
        f = CubicHermiteSpline([x1, x2], [y1, y2], dy)
        xnew = np.linspace(x1, x2, int((x2 - x1) / 0.01) + 1)
        ynew = f(xnew)
        xcoords.extend(xnew)
        ycoords.extend(ynew)

    elif idx == 9:
        def cubic_interpolation(x0, y0, m0, x1, y1, m1, xs):
            middle = (x0 + x1) / 2
            a = np.sqrt(3) / 2 * (x1 - x0)
            p = 3 / 2 * np.sqrt(3) / np.power(a, 3) * height
            return np.array([p * (x - middle) * (x - a - middle) * (x + a - middle) - 5.25 for x in xs])

        x1, x2 = conex - 30, conex
        y1, y2 = -5.25 + height, -5.25 - height
        xnew = np.linspace(x1, x2, int((x2 - x1) / 0.01) + 1)
        ynew = cubic_interpolation(x1, y1, 0, x2, y2, 0, xnew)
        xcoords.extend(xnew)
        ycoords.extend(ynew)

        x1, x2 = conex, conex + 15
        y1, y2 = -5.25 - height, -5.25
        dy = [0, 0]
        f = CubicHermiteSpline([x1, x2], [y1, y2], dy)
        xnew = np.linspace(x1, x2, int((x2 - x1) / 0.01) + 1)
        ynew = f(xnew)
        xcoords.extend(xnew)
        ycoords.extend(ynew)

    elif idx % 2 == 0:
        def cubic_interpolation(x0, y0, m0, x1, y1, m1, xs):
            middle = (x0 + x1) / 2
            a = np.sqrt(3) / 2 * (x1 - x0)
            p = -3 / 2 * np.sqrt(3) / np.power(a, 3) * height
            return np.array([p * (x - middle) * (x - a - middle) * (x + a - middle) - 5.25 for x in xs])

        x1, x2 = conex - 30, conex
        y1, y2 = -5.25 - height, -5.25 + height
        xnew = np.linspace(x1, x2, int((x2 - x1) / 0.01) + 1)
        ynew = cubic_interpolation(x1, y1, 0, x2, y2, 0, xnew)
        xcoords.extend(xnew)
        ycoords.extend(ynew)

    elif idx % 2 == 1:
        def cubic_interpolation(x0, y0, m0, x1, y1, m1, xs):
            middle = (x0 + x1) / 2
            a = np.sqrt(3) / 2 * (x1 - x0)
            p = 3 / 2 * np.sqrt(3) / np.power(a, 3) * height
            return np.array([p * (x - middle) * (x - a - middle) * (x + a - middle) - 5.25 for x in xs])

        x1, x2 = conex - 30, conex
        y1, y2 = -5.25 + height, -5.25 - height
        xnew = np.linspace(x1, x2, int((x2 - x1) / 0.01) + 1)
        ynew = cubic_interpolation(x1, y1, 0, x2, y2, 0, xnew)
        xcoords.extend(xnew)
        ycoords.extend(ynew)

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
