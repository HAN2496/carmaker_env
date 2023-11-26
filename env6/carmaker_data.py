from shapely import affinity
from carmaker_cone import *
import pandas as pd
from scipy.spatial import KDTree
import pygame
from carmaker_trajectory import Trajectory

class Data:
    def __init__(self, road_type, low=True, env_num=2):
        self.road_type = road_type
        self.road = Road(road_type=road_type)
        self.traj = Trajectory(road_type=road_type, low=low)
        self.low = low
        self.env_num = env_num
        self.test_num = 0

        self.do_render = False
        if env_num == 0 and not low:
            self.do_render = True

        if self.road_type == "DLC":
            self.XSIZE, self.YSIZE = 10, 10
        elif self.road_type == "SLALOM" or self.road_type == "SLALOM2":
            self.XSIZE, self.YSIZE = 2, 5

        if self.do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.road.length * self.XSIZE, - self.road.width * self.YSIZE))
            pygame.display.set_caption("B level Environment")

        self._init()

    def _init(self):
        x, y = init_car_pos(road_type=road_type)
        arr = [
            0, 0, x, y, 0, 13.8889,
            0, 0, 0,
            0, 0,
            0, 0, 0, 0, 0, 0
        ]
        self.put_simul_data(arr)

        if self.do_render:
            self.render()
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

    def get_lookahead_traj_rel(self):
        lookahead_sight = [2 * (i + 1) for i in range(5)]
        lookahead_traj_abs = self.traj.find_lookahead_traj(self.carx, self.cary, self.caryaw, lookahead_sight)
        lookahead_traj_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], lookahead_traj_abs).flatten()
        return lookahead_traj_rel

    def get_cones_rel(self, pos):
        """
        pos: [a, b]
        1) a ~ b 사이에 있는 콘의 위치.
        2) if a or b == 0: 현재 차 바로 앞 콘의 위치
        3) if a or b == 3: 현재 차 기준 세 개 앞에 있는 콘의 위치
        """
        if np.shape(pos) != (2, ):
            raise TypeError("Wrong shape of the function get_cones_rel")
        pos = sorted(pos)
        cones_abs = self.road.cone.arr[self.road.cone.arr[:, 0] > self.carx][:3]

    def render(self):
        pass


if __name__ == "__main__":
    road_type = "UTurn"
