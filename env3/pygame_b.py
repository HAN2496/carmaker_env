import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import pygame
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import interp1d
from pygame_low import LowLevelEnv
import time

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

class Road:
    def __init__(self):
        self.road_length = 161
        self.road_width = -20
        self._forbidden_area()
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()

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
            {'start': 0, 'gap': 5, 'cone_dist': 2.23, 'num': 10, 'y_offset': -10},
            {'start': 50, 'gap': 3, 'cone_dist': 2.23, 'num': 5, 'y_offset': -10}, #
            {'start': 64.7, 'gap': 2.7, 'cone_dist': 6.03, 'num': 4, 'y_offset': -8.1},
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2.8, 'num': 5, 'y_offset': -6.485}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 6.8, 'num': 4, 'y_offset': -8.485},
            {'start': 99, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -10.385}, #
            {'start': 111, 'gap': 5, 'cone_dist': 3, 'num': 20, 'y_offset': -10.385}
        ]

        return sections
    def is_car_in_forbidden_area(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)

        if car_shape.intersects(self.forbbiden_area1) or car_shape.intersects(self.forbbiden_area2):
            return 1
        else:
            return 0
    def is_car_colliding_with_cones(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
        for cone in self.cones_shape:
            if car_shape.intersects(cone):
                return 1
        return 0

    def is_car_in_road(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
        if not car_shape.intersects(self.road_boundary):
            return 1
        if not self.road_boundary.contains(car_shape):
            return 1
        return 0

    def is_car_in_cone_area(self, car):
        car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
        if not car_shape.intersects(self.cones_boundary):
            return 1
        if not self.cones_boundary.contains(car_shape):
            return 1
        return 0

class Car:
    def __init__(self):
        self.length = 4.3
        self.width = 1.568
        self.carx = 3
        self.cary = -10
        self.caryaw = 0
        self.carv = 13.8889

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

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()
        self.reset_num = 0
        self.time = 0
        self.test_num = 0

        self.car = Car()
        self.road = Road()

        env_action_num = 1
        env_obs_num = 20
        self.action_space = spaces.Box(low=-1., high=1., shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        low_level_env = LowLevelEnv()
        self.low_level_model = SAC.load("pygame_model.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()

        self.car_dev_before = np.array([0, 0])
        self.traj_data = np.array([[3, -8.0525], [15, -8.0525]])
        self.traj_data = self.make_trajectory()
        self.traj_point = self.find_nearest_point(self.car.carx, self.car.cary, [3*i for i in range(5)])

        pygame.init()
        self.screen = pygame.display.set_mode((self.road.road_length * 10, - self.road.road_width * 10))
        pygame.display.set_caption("B level Environment")


    def _initial_state(self):
        self.time = 0
        self.car.reset_car()
        self.traj_data = np.array([[3, -8.0525], [15, -8.0525]])
        self.traj_data = self.make_trajectory()
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.reset_num += 1
        self.render()
        return self._initial_state()

    def close(self):
        pygame.quit()

    def step(self, action):
        # 기존 action : steering vel -> scalar
        # Policy b action : traj points -> array(list)
        # low_level_obs에 cary, carv 들어가고, car_dev랑 lookahead는 action(신규)에서 가져온 trj 정보로
        done = False
        self.test_num += 1
        sight = np.array([3 * i for i in range(5)])

        traj_lowlevel_abs = self.find_nearest_point(self.car.carx, self.car.cary, sight)
        traj_lowlevel_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, traj_lowlevel_abs).flatten()
        self.low_level_obs = np.concatenate((np.array([self.car.carv]), traj_lowlevel_rel))
        steering_changes = self.low_level_model.predict(self.low_level_obs)

        self.car.move_car(steering_changes[0])

        new_traj_point = self.make_traj_point(action[0])
        self.traj_data = self.make_trajectory(action[0])
        traj_abs = self.find_nearest_point(self.car.carx, self.car.cary, sight)
        self.traj_point = traj_abs
        traj_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, traj_abs).flatten()
        car_dev = self.calculate_dev()
        cones_state = self.road.cones_arr[self.road.cones_arr[:, 0] > self.car.carx][:5]
        cones_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, cones_state).flatten()
        state = np.concatenate((traj_rel, cones_rel)) # <- Policy B의 state

        if self.road.is_car_in_road(self.car) == 1:
            done = True
        reward = self.getReward(car_dev, new_traj_point)
        info = {"carx": self.car.carx, "cary": self.car.cary, "caryaw": self.car.caryaw}

        if self.test_num % 300 == 0:
            self.print_result(reward, car_dev)

        self.time += 0.01
        self.car_dev_before = car_dev
        self.traj_before = traj_abs

        self.render()

        return state, reward, done, info

    def make_traj_point(self, action):
        new_traj_point = np.array([self.car.carx + 12, self.car.cary + action * 3])
        return new_traj_point

    def make_trajectory(self, action=0):
        arr = self.traj_data.copy()

        if action != 0:
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

    def find_nearest_point(self, x0, y0, distances):
        points = []
        for distance in distances:
            filtered_data = self.traj_data[self.traj_data[:, 0] > x0]
            x_diff = np.abs(filtered_data[:, 0] - (x0 + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(filtered_data[nearest_idx])
        return points

    def calculate_dev(self):
        f = interp1d(self.traj_data[:, 0], self.traj_data[:, 1])
        xnew = np.arange(self.traj_data[0][0], self.traj_data[-1][0], 0.01)
        ynew = f(xnew)
        arr = np.array(list(zip(xnew, ynew)))
        distances = np.sqrt(np.sum((arr - [self.car.carx, self.car.cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]
        if arr[dist_index][0] - arr[dist_index - 1][0] == 0:
            devAng2 = np.arctan(np.inf)
        else:
            devAng2 = np.arctan((arr[dist_index][1] - arr[dist_index - 1][1]) / (arr[dist_index][0] - arr[dist_index - 1][0]))
        devAng = - devAng2 - self.car.caryaw
        return devDist, devAng

    def getReward(self, dev, new_traj_point):
        forbidden_reward, cones_reward, car_reward, ang_reward = 0, 0, 0, 0
        traj_point = Point(new_traj_point[0], new_traj_point[1])
        if self.road.forbbiden_area1.intersects(traj_point) or self.road.forbbiden_area2.intersects(traj_point):
            forbidden_reward = -10000
        if self.road.cones_boundary.intersects(traj_point):
            cones_reward = +100
        if self.road.is_car_in_forbidden_area(self.car):
            car_reward = -10000

        e = forbidden_reward + cones_reward + car_reward + ang_reward
        return e

    def to_relative_coordinates(self, carx, cary, caryaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - carx
            dy = point[1] - cary

            rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
            rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        for cone in self.road.cones_shape:
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * 10), int(-y * 10)), 5)

        for trajx, trajy in self.traj_point:
            pygame.draw.circle(self.screen, (0, 128, 0), (trajx * 10, - trajy * 10), 5)

        car_color = (255, 0, 0)

        half_length = self.car.length * 10 / 2.0
        half_width = self.car.width * 10 / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        rotated_corners = []
        for x, y in corners:
            x_rot = x * np.cos(-self.car.caryaw) - y * np.sin(-self.car.caryaw) + self.car.carx * 10
            y_rot = x * np.sin(-self.car.caryaw) + y * np.cos(-self.car.caryaw) - self.car.cary * 10
            rotated_corners.append((x_rot, y_rot))

        pygame.draw.polygon(self.screen, car_color, rotated_corners)

        #차량 위치 렌더링
        font = pygame.font.SysFont("arial", 20, True, True)
        text_str = f"X: {round(self.car.carx, 1)}, Y: {round(self.car.cary, 1)}"
        text_surface = font.render(text_str, True, (255, 255, 255))

        # 텍스트 이미지의 위치 계산 (우측 하단)
        text_x = self.road.road_length * 10 - text_surface.get_width() - 10
        text_y = - self.road.road_width * 10 - text_surface.get_height() - 10

        # 렌더링된 이미지를 화면에 그리기
        self.screen.blit(text_surface, (text_x, text_y))

        pygame.display.flip()


    def print_result(self, reward, car_dev):
        print("-" * 50)
        print(
            f"[Time: {round(self.time, 2)}] [Reward: {round(reward, 2)}] [Car dev: {round(car_dev[0], 2), round(car_dev[1], 2)}]")
        print("[Trajectory: ]")
        for point in self.traj_point:
            print(f" [{point[0]:.2f}, {point[1]:.2f}]")

if __name__ == "__main__":
    env = CarEnv()
#    plot(env.road, env.car)
    """
    env.car.carx, env.car.cary, env.car.caryaw = 68, -5.0525, 0
    traj_abs = env.find_lookahead_traj(env.car.carx, env.car.cary, [3 * i for i in range(5)])
    traj_rel = env.to_relative_coordinates(env.car.carx, env.car.cary, env.car.caryaw, traj_abs)
    devdist, devAng = env.calculate_dev()
    print(traj_rel)
    print(f"dst: {devdist}, ang: {devAng}")
    """
    model = SAC("MlpPolicy", env, verbose=0)
    try:
        model.learn(total_timesteps=10000 * 300)
    except KeyboardInterrupt:
        print("Learning interrupted.")
    finally:
        print("Env test Finished.")
