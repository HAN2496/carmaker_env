import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
from carmaker_cone import *
import pandas as pd
from MyBezierCurve import BezierCurve
from scipy.spatial import KDTree
import pygame

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSIZE = 1, 3

class Data:
    def __init__(self, road_type, low_env, check, show=False):
        self.road_type = road_type
        self.low_env = low_env
        self.check = check
        self.show = show

        self.cone = Cone(road_type=road_type)
        self.road = Road(road_type=road_type)
        self.car = Car()
        self.traj = Trajectory(low_env=low_env, road_type=road_type)

        if self.road_type == "DLC":
            self.XSIZE, self.YSIZE = 2, 10
        elif self.road_type == "SLALOM" or self.road_type == "SLALOM2":
            self.XSIZE, self.YSIZE = 10, 10

        if self.show and self.check == 0:
            pygame.init()
            self.screen = pygame.display.set_mode((self.road.road_length * XSIZE, - self.road.road_width * YSIZE))
            pygame.display.set_caption("B level Environment")


        self._init()

    def _init(self):
        self.test_num = 0
        self.time = 0
        self.carx, self.cary = init_car_pos(self.road_type)
        self.caryaw, self.carv = 0, 13.8888889
        self.steerAng, self.steerVel, self.steerAcc = 0, 0, 0
        self.devDist, self.devAng = 0, 0
        self.alHori, self.roll = 0, 0
        self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext = 0, 0, 0, 0, 0, 0
        self.steer = np.array([0, 0, 0])
        self.wheel_steer = np.array([0, 0, 0, 0])
        self.wheel_steer_ext = np.array([0, 0])

        self.devDist, self.devAng = self.traj.calculate_dev(self.carx, self.cary, self.caryaw)

        self.traj._init_traj()

        if self.check == 0 and self.show:
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


        self.devDist, self.devAng = self.traj.calculate_dev(self.carx, self.cary, self.caryaw)

        if self.test_num % 150 == 0 and self.check == 0:
            print(f"Time: {self.time}, Pos : [x: {round(self.carx, 2)}] [y: {round(self.cary, 2)}] Reward : [dist : {round(self.devDist,2)}] [angle : {round(self.devAng, 2)}]")

    def manage_state_low(self):
        lookahead_sight = [2 * (i + 1) for i in range(5)]
        lookahead_traj_abs = self.traj.find_lookahead_traj(self.carx, self.cary, self.caryaw, lookahead_sight)
        lookahead_traj_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], lookahead_traj_abs).flatten()

        if self.road_type == "SLALOM" or "SLALOM2" or "Eight_20m" or "UTurn" or "CRC":
            return np.concatenate(([self.devDist, self.devAng, self.caryaw, self.carv, self.steerAng, self.steerVel,
                         self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext], lookahead_traj_rel))

        ahead_cones = self.cone.cone_arr[self.cone.cone_arr[:, 0] > self.carx][:1]
        behind_cones = self.cone.cone_arr[self.cone.cone_arr[:, 0] <= self.carx][:1]
        closest_cones = np.vstack((behind_cones, ahead_cones))
        closest_cones_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], closest_cones).flatten()

        return np.concatenate(([self.devDist, self.devAng, self.caryaw, self.carv, self.steerAng, self.steerVel,
                         self.rl, self.rr, self.fl, self.fr, self.rr_ext, self.rl_ext], closest_cones_rel, lookahead_traj_rel))

    def manage_reward_low(self):
        dist_reward = abs(self.devDist) * 100
        ang_reward = abs(self.devAng) * 500
        if self.road_type == "DLC" or self.road_type == "SLALOM2":
            col_reward = self.is_car_colliding_with_cone() * 1000
        elif self.road_type == "Eight_20m":
            col_reward = 0
        else:
            col_reward = abs(self.alHori) * 1000

        e = - col_reward - dist_reward - ang_reward

        return e

    def manage_reward_b(self):
        car_shape = Car().shape_car(self.carx, self.cary, self.caryaw)
        trajx, trajy = self.traj.find_traj_point()
        traj_shape = Point(trajx, trajy)

        car_col_reward, traj_col_reward = 0, 0
        car_col_reward -= self.is_car_colliding_with_cone()
        traj_col_reward -= self.is_collding_with_cone(traj_shape)
        traj_col_reward -= self.is_collding_with_forbidden(traj_shape)

        y_reward = - abs(trajy + 10) * 100
        yaw_reward = - abs(self.caryaw) * 300

        e = car_col_reward + traj_col_reward + y_reward + yaw_reward

        return e

    def manage_state_b(self):
        traj_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], self.traj.find_traj_points(self.carx)).flatten()
        cones_abs = self.cone.cone_arr[self.cone.cone_arr[:, 0] > self.carx][:3]
        cones_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], cones_abs).flatten()
        return np.concatenate((traj_rel, cones_rel))

    def is_car_colliding_with_cone(self):
        car_shape = Car().shape_car(self.carx, self.cary, self.caryaw)
        if self.road.cone_boundary.contains(car_shape):
            return 0
        return 1

    def is_collding_with_cone(self, traj_shape):
        for cone in self.cone.cone_shape:
            if traj_shape.intersects(cone):
                return 1
        return 0

    def is_collding_with_forbidden(self, traj_shape):
        if self.road.forbbiden_area1.intersects(traj_shape) or self.road.forbbiden_area2.intersects(traj_shape):
            return 1
        return 0

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        before_cone = (0, 0)
        for idx, cone in enumerate(self.cone.cone_shape):
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * XSIZE), int(-y * YSIZE)), 5)

        for trajx, trajy in self.traj.find_traj_points(self.carx):
            pygame.draw.circle(self.screen, (0, 128, 0), (trajx * XSIZE, - trajy * YSIZE), 5)

        car_color = (255, 0, 0)

        half_length = self.car.length * XSIZE / 2.0
        half_width = self.car.width * YSIZE / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        rotated_corners = []
        for x, y in corners:
            x_rot = x * np.cos(-self.caryaw) - y * np.sin(-self.caryaw) + self.carx * XSIZE
            y_rot = x * np.sin(-self.caryaw) + y * np.cos(-self.caryaw) - self.cary * YSIZE
            rotated_corners.append((x_rot, y_rot))

        pygame.draw.polygon(self.screen, car_color, rotated_corners)

        font = pygame.font.SysFont("arial", 15, True, True)
        x, y = self.traj.find_traj_point()
        text_str = f"Traj : ({round(x, 1)}, {round(y, 1)})"
        text_surface = font.render(text_str, True, (255, 255, 255))

        # 텍스트 이미지의 위치 계산 (우측 하단)
        text_x = self.road.road_length * XSIZE - text_surface.get_width() - XSIZE
        text_y = - self.road.road_width * YSIZE - text_surface.get_height() - YSIZE

        # 렌더링된 이미지를 화면에 그리기
        self.screen.blit(text_surface, (text_x, text_y))

        pygame.display.flip()

class Trajectory:
    def __init__(self, low_env, road_type, point_interval=2, point_num=5):
        self.point_interval = point_interval
        self.point_num = point_num
        self.low_env = low_env
        self.road_type = road_type
        self.check_section = 0

        self.dev = np.array([0, 0])

        self._init_traj()
        self.previous_lookahead_points = []
        self.last_traj_x_dist = 0

    def _init_traj(self):
        if self.low_env:
            self.traj_data = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj.csv").loc[:,
                             ["traj_tx", "traj_ty"]].values
        else:
            x, y = init_car_pos(self.road_type)
            self.b = BezierCurve(x, y, 0.02)
            self.traj_data = self.b.get_xy_points(x)

    def get_last_traj_x(self):
        self.last_traj_x_dist = self.b.curves[-1].nodes[0, -1]
        return self.b.curves[-1].nodes[0, -1]

    def get_last_traj_x_distance(self):
        self.last_traj_x_dist = self.b.curves[-1].nodes[0, -1] - self.b.curves[-1].nodes[0, 0]
        return self.b.curves[-1].nodes[0, -1] - self.b.curves[-1].nodes[0, 0]
    def update_traj(self, carx, action):
        action = action * np.pi / 6
        self.b.add_curve(
            [1, 1, 1, action]
        )
        self.traj_data = self.b.get_xy_points(carx)

    def calculate_dev(self, carx, cary, caryaw):
        if self.low_env:
            return self.calculate_dev_low(carx, cary, caryaw)
        else:
            return self.calculate_dev_b(carx, cary, caryaw)

    def calculate_dev_low(self, carx, cary, caryaw):
        if self.road_type == "DLC":
            return self.calculate_dev_DLC(carx, cary, caryaw)
        elif self.road_type == "SLALOM2":
            return self.calculate_dev_SLALOM2(carx, cary, caryaw)

    def calculate_dev_b(self, carx, cary, caryaw):
        arr = self.b.get_xy_points(carx)
        return calculate_dev([carx, cary, caryaw], arr)

    def calculate_dev_DLC(self, carx, cary, caryaw):
        if carx <= 62:
            return np.array([cary + 10, caryaw])
        elif 75.5 <= carx <= 86.5:
            return np.array([cary + 6.485, caryaw])
        elif 99 <= carx:
            return np.array([cary + 10.385, caryaw])
        else:
            arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:, ["traj_tx", "traj_ty"]].values
            return calculate_dev([carx, cary, caryaw], arr)

    def calculate_dev_SLALOM2(self, carx, cary, caryaw):
        if carx <= 85:
            return np.array([cary + 25, caryaw])
        elif carx >= 400:
            return np.array([cary + 25, caryaw])
        else:
            arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:, ["traj_tx", "traj_ty"]].values
            return calculate_dev([carx, cary, caryaw], arr)

    def calculate_dev_crc(self, carx, cary, caryaw):
        if carx <= 100:
            return np.array([cary, caryaw])
        else:
            norm_yaw = np.mod(caryaw, 2 * np.pi)
            devDist = 30 - np.linalg.norm(np.array([carx, cary]) - np.array([100, 30]))
            devAng = norm_yaw - np.mod(np.arctan2(cary - 30, carx - 100) + np.pi / 2, 2 * np.pi)
            devAng = (devAng + np.pi) % (2 * np.pi) - np.pi
            return np.array([devDist, devAng])

    def find_traj_point(self):
        return self.b.get_last_point()

    def find_traj_points(self, carx):
        points = []
        distances = [self.point_interval * (i+1) for i in range(self.point_num)]
        for distance in distances:
            check_traj = self.b.get_xy_points(carx+distance)
            x_diff = np.abs(check_traj[:, 0] - (carx + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(check_traj[nearest_idx])
        return np.array(points)

    def find_lookahead_traj(self, x, y, yaw, distances):
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

        return np.array(result_points)

    """
    def find_lookahead_traj(self, carx, cary, caryaw, distances):
        car_position = np.array([carx, cary])
        distances = np.array(distances)
        result_points = []

        # KD-Tree를 사용하여 가장 가까운 점 찾기
        min_idx = np.argmin(np.sum((self.traj_data - np.array([carx, cary])) ** 2, axis=1))

        # 각도 차이 계산 최적화
        car_direction = np.array([np.cos(caryaw), np.sin(caryaw)])

        for i, dist in enumerate(distances):
            lookahead_idx = min_idx
            total_distance = 0.0
            forward_point_found = False

            while total_distance < dist and lookahead_idx + 1 < len(self.traj_data):
                next_point = self.traj_data[lookahead_idx + 1]
                vector_to_next_point = next_point - car_position
                angle_diff = np.arctan2(np.cross(car_direction, vector_to_next_point), np.dot(car_direction, vector_to_next_point))

                if abs(angle_diff) < np.pi / 2:
                    total_distance += np.linalg.norm(self.traj_data[lookahead_idx + 1] - self.traj_data[lookahead_idx])

                lookahead_idx += 1

            if forward_point_found:
                result_points.append(self.traj_data[lookahead_idx])
            else:
                result_points.append(self.traj_data[lookahead_idx])

        self.previous_lookahead_points = result_points
        return np.array(result_points)
    """
    def find_lookahead_traj_straight(self, carx, axis_y, distances):
        return [[carx + distance, axis_y] for distance in distances]

    def find_lookahead_traj_UTurn(self, carx, distances):
        traj = []

        for dist in distances:
            lookahead_point = carx + dist
            if lookahead_point <= 150:
                traj.append([lookahead_point, -3])
            elif lookahead_point >= 158:
                pass
            else:
                pass

        if carx <= 150 - 10 and self.check_section == 0:
            return self.find_lookahead_traj_straight(carx, -3, distances)
        elif carx <= 150 - 10 and self.check_section == 2:
            return self.find_lookahead_traj_straight(carx, -3, -distances)
    def show_traj_data(self):
        plt.scatter(self.traj_data[:, 0], self.traj_data[:, 1])
        plt.title("Trajectory")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.show()

class Test:
    def __init__(self):
        road_type = "UTurn"
        carx, cary, caryaw = 4, -10, 0
        self.data = Data(road_type=road_type, low_env=False, check=0)
        print('--')
        tmp = np.array([
            0, 0, carx, cary,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0
        ])
        self.data.put_simul_data(tmp)
        print('---')
        self.data.traj.update_traj(np.pi/3)
        print(f"carx: {self.data.carx}, cary: {self.data.cary}")
        print('----')
        print(f"here: {self.data.traj.calculate_dev(carx, cary, caryaw)}")
        self.data.traj.traj_data.show_curve()

if __name__ == "__main__":
    road_type = "UTurn"
    #traj = Trajectory(low_env=True, road_type=road_type)
    #print(traj.find_lookahead_traj(10, 0, 0, [0, 2, 4, 8, 10]))
    test=Test()
