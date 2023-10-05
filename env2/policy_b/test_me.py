import numpy as np
#import pygame
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
import pandas as pd

road = np.array([(0,0), (500, 0), (500, -8.75), (0, -8.75)])
road_polygon = Polygon(road)
x, y = road_polygon.exterior.xy
plt.fill(x, y, alpha=0.3)
plt.plot(x, y, label='Road')
for idx, (x, y) in enumerate(road):
    if idx %2 == 0:
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    if idx %2 == 1:
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')

cone = np.array([[100 + 30 * i, -5.25] for i in range(10)])
plt.scatter(cone[:, 0], cone[:, 1], label='Cone')

for idx, (x, y) in enumerate(cone):
    if idx %2 == 0:
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    if idx %2 == 1:
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')


traj_data = pd.read_csv("traj_slalom.csv").loc[:, ["traj_tx", "traj_ty"]].values
plt.scatter(traj_data[:, 0], traj_data[:, 1], label='Trajectory')

plt.xlabel('m')
plt.ylabel('m')
plt.title("Cone Position in SLALOM")
plt.legend(loc='upper right')
#plt.axis('equal')
plt.show()