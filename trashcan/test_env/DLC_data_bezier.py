import numpy as np
from DLC_cone import Road, Car, Cone
import pygame
from MyBezierCurve import BezierCurve
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

XSIZE, YSIZE = 10, 10

class Data:
    def __init__(self, point_interval=2, point_num=5, check=1, show=True):
        self.point_interval = point_interval
        self.point_num = point_num
        self.sight = self.point_interval * (self.point_num - 1)
        self.check = check
        self.show = show

        self.cone = Cone()
        self.road = Road()
        self.car = Car()


        if self.show and self.check == 0:
            pygame.init()
            self.screen = pygame.display.set_mode((self.road.road_length * XSIZE, - self.road.road_width * YSIZE))
            pygame.display.set_caption("B level Environment")

        self._init()

    def _init(self):
        self.test_num = 0
        self.time = 0
        self.carx, self.cary, self.caryaw, self.carv = 2.1976, -10, 0, 13.8889
        self.steerAng, self.steerVel, self.steerAcc = 0, 0, 0
        self.devDist, self.devAng = 0, 0
        self.alHori, self.roll = 0, 0

        self.bcurve = BezierCurve(0.001)
        self.bcurve.update([self.carx, self.cary, self.caryaw], [5, 5, 5, 0, 0])
        self.traj_arr = self.bcurve.get_xy_points()

        if self.check == 0:
            self.render()

    def put_simul_data(self, arr):
        self.simul_data = arr
        self.test_num = arr[0]
        self.time = arr[1]
        self.carx, self.cary, self.caryaw, self.carv = arr[2:6]
        self.steerAng, self.steerVel, self.steerAcc = arr[6:9]
        self.devDist, self.devAng = arr[9:11]
        self.alHori, self.roll = arr[11:13]

    def make_trajectory(self, action):
        self.bcurve.update([self.carx, self.cary, self.caryaw], [5, 5, 5, action[0], action[1]])
        self.traj_arr = np.concatenate((self.traj_arr, self.bcurve.get_xy_points()), axis=0)

    def find_traj_points(self):
        distances = np.array([self.point_interval * i for i in range(self.point_num)])

        forward_points = self.traj_arr[self.traj_arr[:, 0] > self.carx - self.point_interval/10]  # assuming forward means a larger 'x' value

        data_dist = np.sqrt(np.sum((forward_points - [self.carx, self.cary]) ** 2, axis=1))

        nearest_points = []
        for dist in distances:
            abs_diff = np.abs(data_dist - dist)
            idx = np.argmin(abs_diff)
            nearest_points.append(forward_points[idx])

        return np.array(nearest_points)

    def render(self):
        pass


class Test:
    def __init__(self):
        self.data = Data()
        arr = np.array([1, 1, 20, -10, 0.1, 13.8889, 0, 0, 0, 0, 0, 0, 0])
        self.data.put_simul_data(arr)
        self.data.bcurve.show_curve()

if __name__ == "__main__":
    data = Data()
    arr = np.array([10] * 17)
    data.put_simul_data(arr)
    test = Test()


