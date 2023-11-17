import numpy as np
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt

def uturn_course2(carx):

def find_lookahead_traj_straight(carx, cary, caryaw, axis_y, distances):
    return [[carx + distance, axis_y] for distance in distances]

x, y, yaw, axis = 10 ,0 ,0, 0
distances = [0, 2, 4, 6, 7]
a = find_lookahead_traj_straight(x, y, yaw, axis, distances)
print(a)