import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

df = pd.read_csv('datasets_traj.csv').loc[:, ["traj_tx", "traj_ty"]].values

plt.scatter(df[:, 0], df[:, 1])
plt.show()

def entry_section_function(x):
    return -165.53 + 6.59*x - 0.072*x**2 + 0.00025*x**3


def circular_section(theta, radius, center_x, center_y):
    return center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)

radius = 20
center_x, center_y = 120, 30
circular_section = np.array([circular_section(t, radius, center_x, center_y) for t in np.linspace(0, 2*np.pi, 100)])

def circular_motion(theta_start, theta_end, center, radius, num_points):
    thetas = np.linspace(theta_start, theta_end, num_points)
    return np.array([[center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)] for theta in thetas])

# Example usage
center = (120.12, 26.69)
radius = 21.44
theta_start = 0
theta_end = 2 * np.pi  # Full circle
num_points = 1000  # Number of points to generate

circular_section = circular_motion(theta_start, theta_end, center, radius, num_points)



straight_line = np.array([[x, -5.5] for x in np.linspace(0, 100, int((100 - 0) / 0.01) + 1)])
entry_section = np.array([[x, entry_section_function(x)] for x in np.linspace(100.29, 120.27, int((120.27 - 100.29) / 0.01) + 1)])


total_traj = np.vstack((straight_line, entry_section, circular_section))
plt.scatter(total_traj[:, 0], total_traj[:, 1], marker='.')
plt.show()
