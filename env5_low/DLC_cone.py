import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSZIE = 10, 10

def plot(cone, road, car):
    #plt.figure(figsize=(10, 5))

    # Plot forbidden areas
    plt.plot(*road.cones_boundary.exterior.xy, label="Cone Boundary", color='red')
    #    plt.plot(*road.forbbiden_area1.exterior.xy, label="Forbidden Area 1", color='red')
    #    plt.plot(*road.forbbiden_area2.exterior.xy, label="Forbidden Area 2", color='blue')
#    plt.plot(*road.road_boundary.exterior.xy, label="ROAD BOUNDARY", color='green')

    cones_x = cone.cones_arr[:, 0]
    cones_y = cone.cones_arr[:, 1]
    plt.scatter(cones_x, cones_y, s=10, color='orange', label="Cones")

    car_shape = car.shape_car(4, -9, 0)
    plt.plot(*car_shape.exterior.xy, color='blue', label="Car")
    plt.scatter(car.carx, car.cary, color='blue')

    if road.cones_boundary.contains(car_shape):
        print("IN BOUNDARY")
    else:
        print("COLLISION")

    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.gca().invert_yaxis()
    plt.title('Car, Forbidden Areas and Cones')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

class Cone:
    def __init__(self):
        self.cone_r = 0.2
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()
    def create_cone_shape(self):
        sections = self.create_DLC_cone()
        cones = []
        for section in sections:
            for i in range(section['num']):  # Each section has 5 pairs
                x_base = section['start'] + section['gap'] * i
                y1 = section['y_offset'] - section['cone_dist'] / 2
                y2 = section['y_offset'] + section['cone_dist'] / 2
                cone1 = Point(x_base, y1).buffer(0.2)  # 반지름이 0.2m인 원 생성
                cone2 = Point(x_base, y2).buffer(0.2)  # 반지름이 0.2m인 원 생성
                cones.extend([cone1, cone2])

        return np.array(cones)

    def create_cone_arr(self):
        sections = self.create_DLC_cone()
        cones = []
        for section in sections:
            for i in range(section['num']):  # Each section has 5 pairs
                x_base = section['start'] + section['gap'] * i
                y1 = section['y_offset'] - section['cone_dist'] / 2
                y2 = section['y_offset'] + section['cone_dist'] / 2
                cones.extend([[x_base, y1], [x_base, y2]])

        return np.array(cones)

    def create_DLC_cone(self):
        sections = [
            {'start': -5, 'gap': 5, 'cone_dist': 2.23, 'num': 11, 'y_offset': -10},
            {'start': 50, 'gap': 3, 'cone_dist': 2.23, 'num': 5, 'y_offset': -10}, #
            {'start': 64.7, 'gap': 2.7, 'cone_dist': 6.03, 'num': 4, 'y_offset': -8.1},
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2.8, 'num': 5, 'y_offset': -6.485}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 6.8, 'num': 4, 'y_offset': -8.485},
            {'start': 99, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -10.385}, #
            {'start': 111, 'gap': 5, 'cone_dist': 3, 'num': 20, 'y_offset': -10.385}
        ]
        return sections

class Road:
    def __init__(self):
        self.road_length = 161
        self.road_width = -20
        self._forbidden_area()
        self.cone = Cone()

    def _forbidden_area(self):
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

    def is_car_in_forbidden_area(self, car_shape):
        if car_shape.intersects(self.forbbiden_area1) or car_shape.intersects(self.forbbiden_area2):
            return 1
        else:
            return 0
    def is_car_colliding_with_cones(self, car_shape):
        for cone in self.cone.cones_shape:
            if car_shape.intersects(cone):
                return 1
        return 0

    def is_car_in_road(self, car_shape):
        if not car_shape.intersects(self.road_boundary):
            return 1
        if not self.road_boundary.contains(car_shape):
            return 1
        return 0

class Car:
    def __init__(self, carx=3, cary=-10, caryaw=0, carv=13.8889):
        self.length = CARLENGTH
        self.width = CARWIDTH
        self.carx = carx
        self.cary = cary
        self.caryaw = caryaw
        self.carv = carv

    def reset_car(self):
        self.carx = 3
        self.cary = -10
        self.caryaw = 0

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
    cone =Cone()
    road = Road()
    car = Car()
    carx = 10
    cones_abs = cone.cones_arr[cone.cones_arr[:, 0] > carx][:2]
    print(cones_abs)
    plot(cone, road, car)
