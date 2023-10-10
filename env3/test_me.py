import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def to_relative_coordinates(self, x, y, yaw, abs_coords):
    relative_coords = []

    for point in abs_coords:
        dx = point[0] - x
        dy = point[1] - y

        rotated_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
        rotated_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

        relative_coords.append((rotated_x, rotated_y))

    return np.array(relative_coords)

def calculate_dev(traj_data, carx, cary, caryaw):
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

road_type = "DLC"
traj_data = pd.read_csv(f"datafiles/{road_type}/datasets_traj.csv").loc[:, ["traj_tx", "traj_ty"]].values
carx, cary, caryaw = 70, -11, 0
a = calculate_dev(traj_data, carx, cary, caryaw)
print(f"Dist: {a[0]}, Ang: {a[1]}")
