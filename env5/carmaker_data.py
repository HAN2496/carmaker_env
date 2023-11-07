import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
from carmaker_cone import *

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSZIE = 10, 10

class Data:
    def __init__(self, road_type, low_env):
        self.road_type = road_type

        self.cone = Cone(road_type=road_type)
        self.road = Road(road_type=road_type)
        self.car = Car()

        self.traj_data = Trajectory(low_env=low_env)

        if self.road_type == "DLC":
            self.XSIZE, self.YSIZE = 2, 10
        elif self.road_type == "SLALOM" or self.road_type == "SLALOM2":
            self.XSIZE, self.YSIZE = 10, 10

        self._init_sim()

    def _init_sim(self):
        self.test_num = 0
        self.time = 0
        self.carx, self.cary, self.caryaw, self.carv = 2.9855712, -10, 0, 13.8888889
        self.steerAng, self.steerVel, self.steerAcc = 0, 0, 0
        self.devDist, self.devAng = 0, 0
        self.alHori, self.roll = 0, 0
        self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext = 0, 0, 0, 0, 0, 0
        self.steer = np.array([0, 0, 0])
        self.wheel_steer = np.array([0, 0, 0, 0])
        self.wheel_steer_ext = np.array([0, 0])

    def put_simul_data(self, arr):
        self.simul_data = arr
        self.test_num = arr[0]
        self.time = arr[1]
        self.carx, self.cary, self.caryaw, self.carv = arr[2:6]
        self.steerAng, self.steerVel, self.steerAcc = arr[6:9]
        self.alHori, self.roll = arr[9:11]
        self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext = arr[11:]

        self.steer = arr[6:9]
        self.wheel_steer = arr[11:15]
        self.wheel_steer_ext = arr[15:]

    def manage_state_low(self):
        return np.array([self.devDist, self.devAng, self.caryaw, self.carv, self.steerAng, self.steerVel,
                         self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext])

    def calculate_dev(self, carx, cary, caryaw):
        traj = self.traj_data
        distances = np.sqrt(np.sum((traj - [carx, cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]

        dx1 = traj[dist_index + 1][0] - traj[dist_index][0]
        dy1 = traj[dist_index + 1][1] - traj[dist_index][1]

        dx2 = traj[dist_index][0] - traj[dist_index - 1][0]
        dy2 = traj[dist_index][1] - traj[dist_index - 1][1]

        # 분모가 0이 될 수 있는 경우에 대한 예외처리
        if dx1 == 0:
            devAng1 = np.inf if dy1 > 0 else -np.inf
        else:
            devAng1 = dy1 / dx1

        if dx2 == 0:
            devAng2 = np.inf if dy2 > 0 else -np.inf
        else:
            devAng2 = dy2 / dx2

        devAng = - np.arctan((devAng1 + devAng2) / 2) - caryaw
        return np.array([devDist, devAng])

class Trajectory:
    def __init__(self, low_env, point_interval=2, point_num=5):
        self.point_interval = point_interval
        self.point_num = point_num
        self.low_env = low_env
        self.traj_points = np.array([[2.9855712, -10]])
        self.traj_data = np.array([[2.9855712, -10]])


    def calculate_dev(self, carx, cary, caryaw):
        arr = np.array(self.traj_data)
        distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]

        dx1 = arr[dist_index + 1][0] - arr[dist_index][0]
        dy1 = arr[dist_index + 1][1] - arr[dist_index][1]

        dx2 = arr[dist_index][0] - arr[dist_index - 1][0]
        dy2 = arr[dist_index][1] - arr[dist_index - 1][1]

        # 분모가 0이 될 수 있는 경우에 대한 예외처리
        if dx1 == 0:
            devAng1 = np.inf if dy1 > 0 else -np.inf
        else:
            devAng1 = dy1 / dx1

        if dx2 == 0:
            devAng2 = np.inf if dy2 > 0 else -np.inf
        else:
            devAng2 = dy2 / dx2

        devAng = - np.arctan((devAng1 + devAng2) / 2) - caryaw
        return np.array([devDist, devAng])

    def find_lookahead_traj(self, x, y, distances):
        distances = np.array(distances)
        result_points = []

        min_idx = np.argmin(np.sum((self.traj_data - np.array([x, y])) ** 2, axis=1))

        for dist in distances:
            lookahead_idx = min_idx
            total_distance = 0.0
            while total_distance < dist and lookahead_idx + 1 < len(self.traj_data):
                total_distance += np.linalg.norm(self.traj_data[lookahead_idx + 1] - self.traj_data[lookahead_idx])
                lookahead_idx += 1

            if lookahead_idx < len(self.traj_data):
                result_points.append(self.traj_data[lookahead_idx])
            else:
                result_points.append(self.traj_data[-1])

        return result_points

class Test:
    def __init__(self):
        road_type = "DLC"
        self.data = Data(road_type=road_type, low_env=True)
        tmp = np.zeros(17)
        self.data.put_simul_data(tmp)
        print(self.data.carx)

if __name__ == "__main__":
    road_type = "DLC"
    traj = Trajectory(low_env=True)
    test=Test()
