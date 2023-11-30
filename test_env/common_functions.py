import numpy as np
import sys

def check_car_pos(car_pos):
    if np.size(car_pos) != 3:
        print("Error: Car Pos array have to include carx, cary, caryaw")
        sys.exit(1)

def to_relative_coordinates(arr, car_pos):
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

def calculate_dev(car_pos, traj_data):
    check_car_pos(car_pos)
    carx, cary, caryaw = car_pos
    arr = np.array(traj_data)
    distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
    dist_index = np.argmin(distances)
    devDist = distances[dist_index]

    dx = arr[dist_index][0] - arr[dist_index - 1][0]
    dy = arr[dist_index][1] - arr[dist_index - 1][1]

    if dx == 0:
        devAng = np.inf if dy > 0 else -np.inf
    else:
        devAng = dy / dx

    devAng = - np.arctan(devAng) - caryaw
    return np.array([devDist, devAng])

def make_traj_point(car_pos, action):
    check_car_pos(car_pos)
    carx, cary, caryaw = car_pos
    theta = action * 0.1 + caryaw
    new_traj_point = np.array([carx + 8 * np.cos(theta),
                               cary + 8 * np.sin(theta)])
    return new_traj_point

def check_collision(shape1, shape2):
    if shape1.intersects(shape2):
        return 1
    return 0
