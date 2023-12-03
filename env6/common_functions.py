import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
import pandas as pd

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
SLALOM2_Y = -25

GRAY = (128, 128, 128)
RED = (255, 0, 0)
ORANGE = (255, 144, 0)
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)

"""
Cone 관련 함수들
"""
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

"""
DATA 관련 함수들
"""

def init_car_pos(road_type):
    if road_type == "CRC":
        return np.array([2.36088498, -5.5])
    elif road_type == "DLC":
        return np.array([2.9855712, -10])
    elif road_type == "SLALOM2" or "SLALOM":
        return np.array([2.9855712, -25.0])
    elif road_type == "UTurn":
        return np.array([2.3609321776837224, -3.0])
    elif road_type == "Eight_20m":
        return np.array([0, 6.27E-06])

def check_car_pos(car_pos):
    if np.size(car_pos) != 3:
        raise TypeError("Error: Car Pos array have to include carx, cary, caryaw")

def to_relative_coordinates(car_pos, arr):
    check_car_pos(car_pos)
    relative_coords = []
    carx, cary, caryaw = car_pos

    for point in arr:
        dx = point[0] - carx
        dy = point[1] - cary

        rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
        rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

        relative_coords.append((rotated_x, rotated_y))

    return np.array(relative_coords)

"""
Trajectory 관련 함수
"""
def calculate_dev(car_pos, traj_data):
    check_car_pos(car_pos)
    carx, cary, caryaw = car_pos

    norm_yaw = np.mod(caryaw, 2 * np.pi)

    arr = np.array(traj_data)
    distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
    dist_index = np.argmin(distances)
    devDist = distances[dist_index]

    dx = arr[dist_index][0] - arr[dist_index - 1][0]
    dy = arr[dist_index][1] - arr[dist_index - 1][1]
    path_ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    devAng = norm_yaw - path_ang
    devAng = (devAng + np.pi) % (2 * np.pi) - np.pi
    return np.array([devDist, devAng])

def calculate_dev_low(car_pos, traj_data):
    check_car_pos(car_pos)
    carx, cary, caryaw = car_pos
    arr = np.array(traj_data)
    distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
    dist_index = np.argmin(distances)
    devDist = distances[dist_index]

    dx1 = arr[dist_index + 1][0] - arr[dist_index][0]
    dy1 = arr[dist_index + 1][1] - arr[dist_index][1]

    dx2 = arr[dist_index][0] - arr[dist_index - 1][0]
    dy2 = arr[dist_index][1] - arr[dist_index - 1][1]

    # 분모가 0이 될 수 있는 경우에 대한 예외처리
    if dx1 == 0:
        devAng1 = np.inf if dy1 > 0 else -np.inf
    else:
        devAng1 = dy1 / dx1

    if dx2 == 0:
        devAng2 = np.inf if dy2 > 0 else -np.inf
    else:
        devAng2 = dy2 / dx2

    devAng = - np.arctan((devAng1 + devAng2) / 2) - caryaw
    return np.array([devDist, devAng])