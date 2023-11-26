import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSZIE = 10, 10

def create_SLALOM_cone_arr(y_middle):
    cones = []
    more_before_cone = np.array([[-30, y_middle]])
    for i in range(10):
        sign = (i % 2) * 2 - 1  # [-1 1]
        cone = np.array([100 + 30 * i, y_middle])
        cones.append(cone)
    further_cones = np.array([[600, y_middle]])
    cones = np.concatenate((more_before_cone, cones, further_cones), axis=0)
    return cones

def create_SLALOM_cone_arr_sign():
    cones = []
    more_before_cone = np.array([[-30, -7, +1]])
    # 좌측이 -1, 우측이 +1 (yaw가 +일때 시계방향으로 회전함) 1
    for i in range(10):
        sign = (i % 2) * 2 - 1  # [-1 1]
        cone = np.array([100 + 30 * i, - 10 - sign * DIST_FROM_AXIS, (i % 2) * 2 - 1])
        cones.append(cone)
    further_cones = np.array(
        [[800 + 30 * int(i / 2), -10 + ((i % 2) - 0.5) * 2 * 3, (i % 2) * 2 - 1] for i in range(10)])
    cones = np.concatenate((more_before_cone, cones, further_cones), axis=0)
    return cones

def create_DLC_cone_arr():
    sections = [
        {'start': -5, 'gap': 5, 'cone_dist': 2.23, 'num': 11, 'y_offset': -10},
        {'start': 50, 'gap': 3, 'cone_dist': 2.23, 'num': 5, 'y_offset': -10},  #
        {'start': 64.7, 'gap': 2.7, 'cone_dist': 6.03, 'num': 4, 'y_offset': -8.1},
        {'start': 75.5, 'gap': 2.75, 'cone_dist': 2.8, 'num': 5, 'y_offset': -6.485},  #
        {'start': 89, 'gap': 2.5, 'cone_dist': 6.8, 'num': 4, 'y_offset': -8.485},
        {'start': 99, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -10.385},  #
        {'start': 111, 'gap': 5, 'cone_dist': 3, 'num': 20, 'y_offset': -10.385}
    ]
    cones = []

    for section in sections:
        for i in range(section['num']):  # Each section has 5 pairs
            x_base = section['start'] + section['gap'] * i
            y1 = section['y_offset'] - section['cone_dist'] / 2
            y2 = section['y_offset'] + section['cone_dist'] / 2
            cones.extend([[x_base, y1], [x_base, y2]])

    return np.array(cones)

