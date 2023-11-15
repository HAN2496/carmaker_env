import numpy as np
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt

a = np.array([0, 0])
b = np.array([3, 4])
c = np.sqrt((a - b) ** 2)
print(c)


def make_semicircle(x0, y0, r_in, r_out, direction):
    circle_out = Point(x0, y0).buffer(r_out)
    circle_in = Point(x0, y0).buffer(r_in)
    rec = Polygon([[x0 - r_out, y0], [x0 + r_out, y0], [x0 + r_out, y0 - direction * r_out],
                   [x0 - r_out, y0 - direction * r_out]])
    return circle_out.difference(circle_in).difference(rec)

x0, y0 = 0, 0
r_in, r_out = 1, 10
direction = -1

def calculate_dev_crc(carx, cary, caryaw):
    norm_yaw = np.mod(caryaw, 2 * np.pi)
    #반시계가 +임.
    devDist = np.linalg.norm(np.array([carx, cary]) - np.array([100, 30]))
    devAng = np.mod(np.arctan2(cary - 30, carx - 100) + np.pi / 2, 2 * np.pi) - norm_yaw
    return devDist, devAng

x = 100
y = 60
yaw = np.pi * 5

print(calculate_dev_crc(x, y, yaw))
