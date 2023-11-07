import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSZIE = 10, 10

def plot(cone_arr):
    plt.scatter(cone_arr[:, 0], cone_arr[:, 1], label="Cone")

    plt.legend()
    plt.axis('equal')
    plt.show()

def create_SLALOM_cone_arr():
    cones = []
    more_before_cone = np.array([[-30, -10]])
    for i in range(10):
        sign = (i % 2) * 2 - 1  # [-1 1]
        cone = np.array([100 + 30 * i, -10])
        cones.append(cone)
    further_cones = np.array([[600, -10]])
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

class Cone:
    def __init__(self, road_type):
        self.cone_r = CONER
        self.road_type = road_type
        if road_type == "DLC":
            self.cone_arr = create_DLC_cone_arr()
        elif road_type == "SLALOM":
            self.cone_arr = create_SLALOM_cone_arr()
        elif road_type == "SLALOM2":
            self.cone_arr = create_SLALOM_cone_arr_sign()
        else:
            self.cone_arr = create_SLALOM_cone_arr_sign()
        self.cone_shape = self.create_cone_shape()

    def create_cone_shape(self):
        cone_shape = []
        if self.road_type == "SLALOM2":
            for i, j, k in self.cone_arr:
                cone = Point(i, j).buffer(self.cone_r)
                cone_shape.append(cone)

            return np.array(cone_shape)
        else:
            for i, j in self.cone_arr:
                cone = Point(i, j).buffer(self.cone_r)
                cone_shape.append(cone)
            return np.array(cone_shape)

class Road:
    def __init__(self, road_type):
        self.road_type = road_type
        if road_type == "DLC":
            self.create_DLC_road()
        elif road_type == "SLALOM":
            self.create_SLALOM_road()

    def create_SLALOM_road(self):
        self.road_length = 500
        self.road_width = -20
        vertices1 = [[100 + 30 * i, - 10 - DIST_FROM_AXIS - 7 * (i % 2 - 1)] for i in range(10)]
        vertices1 = np.array(vertices1 + [[385, -7 - DIST_FROM_AXIS], [510, -7 - DIST_FROM_AXIS], [510, 15], [0, 15],
                                          [0, -7 - DIST_FROM_AXIS], [85, -7 - DIST_FROM_AXIS], [100, - DIST_FROM_AXIS - 3]])
        vertices2 = [[100 + 30 * i, -10 + DIST_FROM_AXIS - 7 * (i % 2)] for i in range(10)]
        vertices2 = np.array(vertices2 + [[385, -13 + DIST_FROM_AXIS], [510, -13 + DIST_FROM_AXIS], [510, -35], [0, -35],
                                          [0, -13 + DIST_FROM_AXIS], [85, -13 + DIST_FROM_AXIS], [100, -10 + DIST_FROM_AXIS]])
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbidden_line1 = vertices1
        self.forbidden_line2 = vertices2

        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])

    def create_DLC_road(self):
        self.road_length = 161
        self.road_width = -20
        vertices1 = [
            (0, -8.885), (62, -8.885), (62, -5.085), (99, -5.085), (99, -8.885),
            (161, -8.85), (161, 0), (0, 0), (0, -8.885)
        ]
        vertices2 = [
            (0, -11.885), (75.5, -11.885), (75.5, -7.885), (86.5, -7.885),
            (86.5, -11.885), (161, -11.885), (161, -20), (0, -20), (0, -11.885)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])
        self.cones_boundary = Polygon(
            [(0, -8.885), (62, -8.885), (62, -5.085), (99, -5.085), (99, -8.885), (161, -8.885),
             (161, -11.885), (86.5, -11.885), (86.5, -7.885), (75.5, -7.885), (75.5, -11.115), (0, -11.115)
        ])

    def plot_road(self):
        plt.plot(*self.road_boundary.exterior.xy, label='road boundary', color='blue')
        plt.plot(*self.forbbiden_area1.exterior.xy, label='forbidden area', color='red')
        plt.plot(*self.forbbiden_area2.exterior.xy, color='red')
        plt.plot(*self.cones_boundary.exterior.xy, label="Cone Boundary", color='orange')
        plt.fill(*self.cones_boundary.exterior.coords.xy, color='orange', alpha=0.5)
        plt.legend()
        plt.show()

class Car:
    def __init__(self):
        self.length = CARLENGTH
        self.width = CARWIDTH
        self.reset_car()

    def reset_car(self):
        self.carx = 2.9855712
        self.cary = -10
        self.caryaw = 0
        self.carv = 13.8888889

    def move_car(self, angle):
        angle = angle[0]
        self.caryaw += angle * 0.01
        self.carx += np.cos(self.caryaw) * self.carv * 0.01
        self.cary += np.sin(self.caryaw) * self.carv * 0.01

    def shape_car(self, carx, cary, caryaw):
        half_length = self.length / 2.0
        half_width = self.width / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        car_shape = Polygon(corners)
        car_shape = affinity.rotate(car_shape, caryaw, origin='center', use_radians=False)
        car_shape = affinity.translate(car_shape, carx, cary)

        return car_shape


if __name__ == "__main__":
    road_type = "DLC"
    cone = Cone(road_type=road_type)
    print(np.shape(cone.cone_shape))
    road = Road(road_type=road_type)
    road.plot_road()
