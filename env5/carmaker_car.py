import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
from common_functions import *
from carmaker_cone import *

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSZIE = 10, 10

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
    road_type = "SLALOM2"
    cone = Cone(road_type=road_type)
    road = Road(road_type=road_type)
    road.plot_road()
