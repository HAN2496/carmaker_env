import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt

cone_r = 0.2
car_width, car_length = 1.568, 4.3
dist_from_axis = (car_width + 1) / 2 + cone_r



def plot(road, car):
    #plt.figure(figsize=(10, 5))

    # Plot forbidden areas
    plt.plot(*road.cones_boundary.exterior.xy, label="Cone Boundary", color='red')
#    plt.plot(*road.forbbiden_area1.exterior.xy, label="Forbidden Area 1", color='red')
#    plt.plot(*road.forbbiden_area2.exterior.xy, label="Forbidden Area 2", color='blue')
    plt.plot(*road.road_boundary.exterior.xy, label="ROAD BOUNDARY", color='green')

    # Plot cones
    cones_x = road.cones_arr[:, 0]
    cones_y = road.cones_arr[:, 1]
    plt.scatter(cones_x, cones_y, s=10, color='orange', label="Cones")

    # Plot the car
    car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
    plt.plot(*car_shape.exterior.xy, color='blue', label="Car")
    plt.scatter(car.carx, car.cary, color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.gca().invert_yaxis()
    plt.title('Car, Forbidden Areas and Cones')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
def is_car_in_forbidden_area(car_shape, road):
    if car_shape.intersects(road.forbbiden_area1) or car_shape.intersects(road.forbbiden_area2):
        return 1
    else:
        return 0
def is_car_colliding_with_cones(car_shape, cone):
    for cone in cone.cones_shape:
        if car_shape.intersects(cone):
            return 1
    return 0

def is_car_in_road(self, car_shape):
    if not car_shape.intersects(self.road_boundary):
        return 1
    if not self.road_boundary.contains(car_shape):
        return 1
    return 0
class Cone:
    def __init__(self):
        self.cone_r = 0.2
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()
        self.middles_arr = self.create_middle_arr()
        self.middles_shape = self.create_middle_shape()
    def create_cone_shape(self):
        cones = []
        dist_from_axis = (car_width + 1) / 2 + cone_r + 1
        for i, j in self.cones_arr:
            cone = Point(i, j).buffer(dist_from_axis)
            cones.append(cone)

        return np.array(cones)

    def create_cone_arr(self):
        road_cones = np.array([[100 + 30 * i, -10] for i in range(10)])
        further_cones = np.array([[800 + 30 * i, -10] for i in range(5)])
        cones = np.concatenate((road_cones, further_cones), axis=0)
        return cones

    def create_middle_arr(self):
        middle = np.array([[85 + 30 * i, -10] for i in range(10)])
        return middle

    def create_middle_shape(self):
        cones = []
        for i, j in self.middles_arr:
            cone = Point(i, j).buffer(0.2)
            cones.append(cone)

        return np.array(cones)
class Road:
    def __init__(self):
        self.road_length = 500
        self.road_width = -20
        self.cone = Cone()
        self.car = Car()
        self._forbidden_area()
    def _forbidden_area(self):
        vertices1 = [(0, 0), (500, 0), (500, -5), (0, -5), (0, 0)]
        vertices2 = [(0, -15), (500, -15), (500, -20), (0, -20), (0, -15)]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
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
        self.length = 4.3
        self.width = 1.568
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
        self.carx = carx
        self.cary = cary
        self.caryaw = caryaw
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
