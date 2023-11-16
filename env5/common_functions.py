import numpy as np
import sys
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import matplotlib.pyplot as plt

def check_car_pos(car_pos):
    if np.size(car_pos) != 3:
        print("Error: Car Pos array have to include carx, cary, caryaw")
        sys.exit(1)

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

def find_lookahead_traj(x, y, distances, traj_data):
    distances = np.array(distances)
    result_points = []

    min_idx = np.argmin(np.sum((traj_data - np.array([x, y])) ** 2, axis=1))

    for dist in distances:
        lookahead_idx = min_idx
        total_distance = 0.0
        while total_distance < dist and lookahead_idx + 1 < len(traj_data):
            total_distance += np.linalg.norm(traj_data[lookahead_idx + 1] - traj_data[lookahead_idx])
            lookahead_idx += 1

        if lookahead_idx < len(traj_data):
            result_points.append(traj_data[lookahead_idx])
        else:
            result_points.append(traj_data[-1])

    return result_points

def normalize_angle(angle):
    y = np.sin(angle)
    x = np.cos(angle)
    return np.arctan2(y, x)

def create_ellipse(center, major_axis, minor_axis, num_points=100):
    angle = np.linspace(0, 2*np.pi, num_points)
    ellipse_x = center[0] + major_axis * np.cos(angle)
    ellipse_y = center[1] + minor_axis * np.sin(angle)
    ellipse_points = np.column_stack([ellipse_x, ellipse_y])
    ellipse = Polygon(ellipse_points)
    return ellipse
def make_semiellipse(x0, y0, major, minor_out, cone_dist, direction):
    circle_out = create_ellipse([x0, y0], major, minor_out)
    circle_in = Point(x0, y0).buffer(cone_dist)
    rec = Polygon([[x0 - major, y0], [x0 + major, y0],
                   [x0 + major, y0 - direction * minor_out], [x0 - major, y0 - direction * minor_out]])

    return circle_out.difference(circle_in).difference(rec)

def init_car_pos(road_type):
    if road_type == "CRC":
        return 2.36088498, -5.5
    elif road_type == "DLC":
        return 2.9855712, -10
    elif road_type == "SLALOM2" or "SLALOM":
        return 2.9855712, -25.0
    elif road_type == "UTurn":
        return 2.3609321776837224, -3.0
    elif road_type == "Eight_20m":
        return 0, 6.27E-06

if __name__ == "__main__":
    x0, y0 = 0, 0
    r_in, r_out = 1, 10
    direction = -1

    a = make_semiellipse(x0, y0, 30, 3, 1, direction)
    b = create_ellipse((x0, y0), 10, 1)

    plt.plot(*a.exterior.xy)
    plt.axis('equal')
    plt.show()
