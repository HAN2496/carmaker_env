from numpy import genfromtxt
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

trackdata = np.load("environment/track/data/trackdata.npz")
cone_position = trackdata["cone_position"]
X = trackdata["X"]
border_left = trackdata["border_left"]
border_right = trackdata["border_right"]

plt.plot(X, np.zeros(X.shape))
plt.plot(X, border_left, "-")
plt.plot(X, border_right, "-")
plt.plot(cone_position, np.zeros(cone_position.shape), "o")

plt.legend(["Reference", "Border left", "Border right", "Cone"])
plt.show()