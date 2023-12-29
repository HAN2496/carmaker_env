import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import pandas as pd
from shapely import affinity
from shapely.ops import unary_union

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

def create_total_line():
    dlc_start = 150
    width1, width2, width3 = 2.4225, 2.975, 3
    offset1, offset2, offset3 = -12.25, -12.25+2.61125, -12.25-0.28875
    up1, up2, up3 = offset1 + width1/2, offset2 + width2/2, offset3 + width3/2
    low1, low2, low3 = offset1 -width1/2, offset2 - width2/2, offset3 - width3/2
    upper_arr1 = np.array([(dlc_start + 0, up1 - CONER), (dlc_start + 62, up1 - CONER), (dlc_start + 62, up2 - CONER),
                       (dlc_start + 99, up2 - CONER), (dlc_start + 99, up3 - CONER), (dlc_start + 200, up3 - CONER)])
    lower_arr1 = np.array([(dlc_start + 0, low1 + CONER), (dlc_start + 75.5, low1 + CONER), (dlc_start + 75.5, low2 + CONER),
                       (dlc_start + 86.5, low2 + CONER), (dlc_start + 86.5, low3 + CONER), (dlc_start + 200, low3 + CONER)])
    straight_dist = (CARWIDTH * 1.1 + 0.25) / 2

    slalom_start = 350
    slalom_y = -12.25
    upper_arr2 = np.array([[slalom_start + 0, slalom_y + straight_dist], [slalom_start + 85, slalom_y + straight_dist]] + \
                      [[slalom_start + 100 + 30 * i, slalom_y - (i % 2 - 1) * 3 - 2 * (i % 2 - 0.5) * np.sqrt(2) * CONER] for i in range(10)] + \
                      [[slalom_start + 400, slalom_y + straight_dist], [slalom_start + 550, slalom_y + straight_dist]])
    lower_arr2 = np.array([[x, y - 2 * straight_dist] for x, y in upper_arr2])

    after_slalom = 550 + 350
    upper_arr3 = np.array([[after_slalom + 0, slalom_y + straight_dist], [after_slalom + 0, slalom_y + straight_dist]])
    upper_arr = np.vstack((upper_arr1, upper_arr2))
    lower_arr = np.vstack((lower_arr1, lower_arr2))
    return upper_arr, lower_arr

def create_Ramp_line(show=False):
    arr = pd.read_csv(f"datafiles/Ramp/datasets_traj.csv").loc[:, ["traj_tx", "traj_ty"]].values
    before_ang = 0
    upper_arr = []
    lower_arr = []
    for idx, (x, y) in enumerate(arr):
        if idx == arr.shape[0] - 1:
            ang = before_ang
        else:
            dx1 = arr[idx + 1][0] - x
            dy1 = arr[idx + 1][1] - y
            front_ang = np.arctan2(dy1, dx1)
            ang = front_ang + np.pi/2
        upper = arr[idx] + np.array([np.cos(ang) * DIST_FROM_AXIS, np.sin(ang) * DIST_FROM_AXIS])
        lower = arr[idx] - np.array([np.cos(ang) * DIST_FROM_AXIS, np.sin(ang) * DIST_FROM_AXIS])
        upper_arr.append(upper)
        lower_arr.append(lower)
        before_ang = ang
    upper_arr = np.array(upper_arr)
    lower_arr = np.array(lower_arr)
    if show:
        plt.scatter(arr[:, 0], arr[:, 1], s=0.1, color='black', label='Traj')
        plt.scatter(upper_arr[:, 0], upper_arr[:, 1], s=0.1, color='red', label='Upper line')
        plt.scatter(lower_arr[:, 0], lower_arr[:, 1], s=0.1, color='blue', label='Lower line')
        plt.legend()
        #plt.xlim([600, 900])
        #plt.ylim([-300, -150])
        plt.show()
    return upper_arr, lower_arr

#create_Ramp_line(show=True)

"""
DATA 관련 함수들
"""
def init_car_pos(road_type):
    if road_type == "CRC":
        return np.array([2.36088498, -5.5, 13.8889])
    elif road_type == "DLC":
        return np.array([2.9855712, -10, 13.8889 * 7/5])
    elif road_type == "SLALOM2" or "SLALOM":
        return np.array([2.9855712, -25.0, 13.8889])
    elif road_type == "UTurn":
        return np.array([2.3609321776837224, -3.0, 5.55556])
    elif road_type == "Eight_20m":
        return np.array([0, 6.27E-06, 5.55556])
    elif road_type == "Ramp":
        return np.array([402.9714655, -12.24999372, 13.8889])

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

def tranform_coords():
    pass
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

def calculate_dev_low(car_pos, traj_data, index=-1):
    check_car_pos(car_pos)
    carx, cary, caryaw = car_pos
    arr = np.array(traj_data)
    if index == -1:
        distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]
    else:
        dist_index = index
        devDist = np.sqrt(np.sum((arr[dist_index] - [carx, cary])**2))

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

def calculate_directions(array):
    array = np.array(array)
    m, n = array.shape
    if n != 2:
        raise ValueError("Array must be of shape (m, 2)")

    directions = np.zeros(m-1)

    for k in range(1, m):
        x_diff = array[k, 0] - array[k-1, 0]
        y_diff = array[k, 1] - array[k-1, 1]
        if x_diff == 0:
            directions[k-1] = np.sign(y_diff)
        else:
            if y_diff == 0:
                directions[k-1] = np.sign(x_diff)
            else:
                directions[k-1] = np.sign(y_diff / x_diff)


    return directions

def dlc_interpolation(pos1, pos2, x):
    x0, y0 = pos1
    x1, y1 = pos2
    a = -2 * (y1 - y0) / (x1 - x0) ** 3
    b = 3 * (y1 - y0) / (x1 - x0) ** 2
    interpolate = []
    for i in np.arange(x0, x1, 0.01):
        interpolate.append([i, a * (x - x0) ** 3 + b * (x - x0) ** 2 + y0])
    interpolate = np.array(interpolate)
    return interpolate
