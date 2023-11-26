import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
from carmaker_cone import *
import pandas as pd
from MyBezierCurve import BezierCurve
from scipy.spatial import KDTree
import pygame


class Data:
    def __init__(self, road_type, low=True, env_num=2):
        self.road_type = road_type
        self.road = Road(road_type=road_type)
        self.low = low
        self.env_num = env_num
        self.test_num = 0

        if env_num == 0 and not low:
            self.render()

    def _init(self):
        x, y = init_car_pos(road_type=road_type)
        arr = [
            0, 0, x, y, 0, 13.8889,
            0, 0, 0,
            0, 0,
            0, 0, 0, 0, 0, 0
        ]
        self.put_simul_data(arr)

    def put_simul_data(self, arr):
        self.simul_data = arr
        self.test_num += 1
        self.time = arr[1]
        self.carx, self.cary, self.caryaw, self.carv = arr[2:6]
        self.steerAng, self.steerVel, self.steerAcc = arr[6:9]
        self.alHori, self.roll = arr[9:11]
        self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext = arr[11:]

        self.steer = arr[6:9]
        self.wheel_steer = arr[11:15]
        self.wheel_steer_ext = arr[15:]


    def render(self):
        pass


if __name__ == "__main__":
    road_type = "UTurn"
