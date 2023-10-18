import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt

cone_r = 0.2
car_width, car_length = 1.568, 4.3
dist_from_axis = (car_width + 1) / 2 + cone_r



def plot(road, cone, car):
    #plt.figure(figsize=(10, 5))

    for cone in cone.cones_arr:
        plt.scatter(cone[0], cone[1], color='green')

    plt.plot(*road.forbbiden_area1.exterior.xy)
    plt.plot(*road.forbbiden_area2.exterior.xy)
    print(cone.cones_forbidden_shape)
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
        self.cones_forbidden_shape = self.create_cone_forbidden_shape()

    def create_cone_shape(self):
        cones = []
        dist_from_axis = (car_width + 1) / 2 + cone_r + 1
        for i, j in self.cones_arr:
            cone = Point(i, j).buffer(dist_from_axis)
            cones.append(cone)

        return np.array(cones)
    def create_cone_forbidden_shape(self):
        cones = []
        for i in range(10):
            sign = ((i-1) % 2) * 2
            dist = 10
            x0, x1 = 30 * i - dist + 100, 30 * i + dist + 100
            vertices = [(x0, -10), (x1, -10), (x1, - sign * 10), (x0, - sign * 10), (x0, -10)]
            cone_area = Polygon(vertices)
            cones.append(cone_area)

        return np.array(cones)

    def create_cone_arr(self):
        cones = []
        before_cones = np.array([[30 * int(i / 2) + 10, -10 + ((i % 2) - 0.5) * 2 * 3] for i in range(6)])
        for i in range(10):
            sign = (i % 2) * 2
            cone1 = np.array([100 + 30 * i, - 10])
            cone2 = np.array([100 + 30 * i, -sign * 10])
            cones.append(cone1)
            cones.append(cone2)
        further_cones = np.array([[800 + 30 * int(i / 2), -10 + ((i % 2) - 0.5) * 2 * 3] for i in range(10)])
        cones = np.concatenate((before_cones, cones, further_cones), axis=0)
        return cones


class Road:
    def __init__(self):
        self.road_length = 500
        self.road_width = -20
        self.cone = Cone()
        self.car = Car()
        self._forbidden_area()
    def _forbidden_area(self):
        vertices1 = [(0, 15), (500, 15), (500, -7), (0, -7), (0, 15)]
        vertices2 = [(0, -13), (500, -13), (500, -35), (0, -35), (0, -13)]
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

if __name__ == "__main__":
    cone = Cone()
    road = Road()
    car = Car()
    plot(road, cone, car)
