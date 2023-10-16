import numpy as np
from SLALOM_cone2 import Road, Car, Cone
import pygame
from scipy.interpolate import interp1d

XSIZE, YSIZE = 2, 5

class Data:
    def __init__(self, point_interval=2, point_num=5, check=1, show=True):
        self.point_interval = point_interval
        self.point_num = point_num
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

        self.traj_data = np.array([self.carx, self.cary])
        self.traj_data = self.make_trajectory(0)

        self.traj_point = np.array([self.carx + self.point_interval * (self.point_num - 1), self.cary])
        self.traj_point_before = self.traj_point
        self.traj_points = self.find_traj_points(self.carx)

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

    def make_traj_point(self, action):
        new_traj_point = np.array([self.carx + 8, self.cary + action * 2])
        return new_traj_point

    def make_trajectory(self, action):
        arr = self.traj_data.copy()

        new_traj_point = self.make_traj_point(action)
        arr = np.vstack((arr, new_traj_point))

        if abs(arr[-2][0] - arr[-1][0]) > 0.01:
            f = interp1d(arr[-2:, 0], arr[-2:, 1])
            xnew = np.arange(arr[-2][0], arr[-1][0], 0.01)
            ynew = f(xnew)
            interpolate_arr = np.column_stack((xnew, ynew))
            new_arr = np.vstack((arr[:-1], interpolate_arr, arr[-1]))
            return new_arr
        else:
            return arr

    def find_traj_points(self, x0):
        points = []
        for i in range(self.point_num):
            distance = i * self.point_interval
            x_diff = np.abs(self.traj_data[:, 0] - (x0 + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(self.traj_data[nearest_idx])
        return np.array(points)

    def calculate_dev(self):
        arr = np.array(self.traj_data)
        distances = np.sqrt(np.sum((arr - [self.carx, self.cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]

        dx = arr[dist_index][0] - arr[dist_index - 1][0]
        dy = arr[dist_index][1] - arr[dist_index - 1][1]

        if dx == 0:
            devAng = np.inf if dy > 0 else -np.inf
        else:
            devAng = dy / dx

        devAng = - np.arctan(devAng) - self.caryaw
        return np.array([devDist, devAng])

    def to_relative_coordinates(self, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - self.carx
            dy = point[1] - self.cary

            rotated_x = dx * np.cos(-self.caryaw) - dy * np.sin(-self.caryaw)
            rotated_y = dx * np.sin(-self.caryaw) + dy * np.cos(-self.caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def manage_state(self, arr, action):
        blevel_action = action[0]
        self.put_simul_data(arr)

        traj_point_new = self.make_traj_point(blevel_action)
        traj_point_new_rel = self.to_relative_coordinates([traj_point_new]).flatten()
        self.traj_point = traj_point_new
        self.traj_data = self.make_trajectory(blevel_action)

        #새로생긴 point가 데이터 중간에 생성되었을수도 있으므로. 뺑뺑돌면 가능은 하겠다.
        self.traj_points = self.find_traj_points(self.carx)
        traj_rel = self.to_relative_coordinates(self.traj_points).flatten()

        cones_abs = self.cone.cones_arr[self.cone.cones_arr[:, 0] > self.carx][:3]
        cones_rel = self.to_relative_coordinates(cones_abs).flatten()

        middle_abs = self.cone.middles_arr[self.cone.middles_arr[:, 0] > self.carx][:2]
        middle_rel = self.to_relative_coordinates(middle_abs).flatten()

        state = np.concatenate((np.array([self.caryaw]), traj_point_new_rel, cones_rel)) # <- Policy B의 state
        reward_argument = {"traj": traj_point_new, "caryaw": self.caryaw}
        info_key = np.array(["time", "x", "y", "yaw", "carv", "ang", "vel", "acc", "devDist", "devAng", "alHori", "roll", "rl", "rr", "fl", "fr"])
        info = {key: value for key, value in zip(info_key, arr[1:])}

        return state, reward_argument, info

    def state_size(self):
        arr = np.zeros((13))
        array, _, _ = self.manage_state(arr, np.array([0]))
        return array.size

    def _init_reward_argument(self):
        return {"traj": self.traj_point, "caryaw": 0}

    def _init_info(self):
        info_key = np.array(["time", "x", "y", "yaw", "carv", "ang", "vel", "acc", "devDist", "devAng", "alHori", "roll", "rl", "rr", "fl", "fr"])
        return {key: value for key, value in zip(info_key, np.zeros(14))}

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        for cone in self.cone.cones_shape:
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * XSIZE), int(-y * YSIZE)), 5)

        for trajx, trajy in self.traj_points:
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

        #차량 위치 렌더링
        font = pygame.font.SysFont("arial", 15, True, True)
        text_str = f"Traj : ({round(self.traj_point[0], 1)}, {round(self.traj_point[1], 1)})"
        text_surface = font.render(text_str, True, (255, 255, 255))

        # 텍스트 이미지의 위치 계산 (우측 하단)
        text_x = self.road.road_length * XSIZE - text_surface.get_width() - XSIZE
        text_y = - self.road.road_width * YSIZE - text_surface.get_height() - YSIZE

        # 렌더링된 이미지를 화면에 그리기
        self.screen.blit(text_surface, (text_x, text_y))

        pygame.display.flip()

class Test:
    def __init__(self):
        self.data = Data()

if __name__ == "__main__":
    data = Data()
    arr = np.array([10] * 17)
    data.put_simul_data(arr)
    test = Test()
    test.data.state_size()


