import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, DQN
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
import pygame

CARLENGTH = 4
CARWIDTH = 10
WINDOWWIDTH = 10
WINDOWLENGTH = 10
CAR_SCALE = 10

class Road:
    def __init__(self):
        self.road_length = 161
        self.road_width = -15
        self._forbidden_area()
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()

    def _forbidden_area(self):
        vertices1 = [
            (0, -6.442), (62, -6.442), (62, -3.221), (99, -3.221), (99, -6.442),
            (161, -6.442), (161, 0), (0, 0), (0, -6.442)
        ]
        vertices2 = [
            (0, -9.663), (75.5, -9.663), (75.5, -6.442), (86.5, -6.442),
            (86.5, -9.663), (161, -9.663), (161, -12.884), (0, -12.884), (0, -9.663)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])
        self.cones_boundary = Polygon(
            [(0, -7.0651), (62, -7.0651), (62, -3.4565), (99, -3.4565), (99, -6.5525), (161, -6.5525),
             (161, -9.5525), (86.5, -9.5525), (86.5, -6.2065), (75.5, -6.2065), (75.5, -9.0399), (0, -9.0399)
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
            {'start': 0, 'gap': 5, 'cone_dist': 1.9748, 'num': 10, 'y_offset': -8.0525},
            {'start': 50, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 64.7, 'gap': 2.7, 'cone_dist': 5.4684, 'num': 4, 'y_offset': -6.3057},
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2, 'num': 5, 'y_offset': -4.8315}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 5.4684, 'num': 4, 'y_offset': -6.3057},
            {'start': 99, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 111, 'gap': 5, 'cone_dist': 1.9748, 'num': 20, 'y_offset': -8.0525}
        ]

        return sections

class Trajectory(gym.Env):
    def __init__(self, carx, cary, caryaw, sight=10):
        super(Trajectory, self).__init__()
        self.cones = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()

        self.road = Road()

        self.road_length = 161
        self.road_width = -15

        self.carx = carx
        self.cary = cary
        self.caryaw = caryaw
        self.sight = sight
        self.traj = np.array([(i, -8.0525) for i in range(161)])

        env_obs_num = 322

        action_y = 3
        action_x = 161
        self.action_space = spaces.MultiDiscrete([action_x, action_y])  # 두 개의 액션 공간, 각각 3개의 액션을 가짐
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.road.road_length * 10, - self.road.road_width * 10))
        pygame.display.set_caption("Car Road Environment")

    def reset(self):
        self.carx = 5  # or any other initial value
        self.cary = -8.0525
        self.caryaw = 0
        self.traj = np.array([(i, -8.0525) for i in range(161)])
        return self.getObservation()

    def move_traj_pos(self, x, y):
        if y == 2:
            self.traj[x][1] += 1
        elif y == 1:
            self.traj[x][1] = 0
        elif y == 0:
            self.traj[x][1] -= 1


    def step(self, action):
        actionx = action[0]
        actiony = action[1]
        self.move_traj_pos(actionx, actiony)
        state = self.getObservation()

        reward = self.getReward()
        done = False
        info = {}

        return state, reward, done, info

    def getObservation(self):
        return self.traj.flatten()

    def getReward(self):
        center_distance = np.abs(self.cary - self.traj[:, 1])
        center_reward = -np.mean(center_distance)

        out_of_path_penalty = 0
        if np.any(self.traj[:, 1] < self.cary - CARWIDTH/2) or np.any(self.traj[:, 1] > self.cary + CARWIDTH/2):
            out_of_path_penalty = -50

        # 3. 경로의 길이에 따른 페널티 (보상을 줄임)
        length_penalty = -np.sum(np.diff(self.traj[:, 1]) ** 2)

        reward = center_reward + out_of_path_penalty + length_penalty
        return reward

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
            {'start': 0, 'gap': 10, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525},
            {'start': 50, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2, 'num': 5, 'y_offset': -4.8315}, #
            {'start': 99, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 121, 'gap': 10, 'cone_dist': 1.9748, 'num': 10, 'y_offset': -8.0525}]
        return sections

    def find_cones_closest_front(self, distance):
        close_cones = self.cones[(self.cones[:, 0] >= self.carx) & (self.cones[:, 0] <= self.carx + distance)]
        return close_cones

    def plotting(self):
        plt.scatter(self.cones[:, 0], self.cones[:, 1], s=10, color='orange', label="Cones")
        plt.scatter(self.traj[:,0], self.traj[:, 1])
        plt.axis("equal")
        plt.show()

    def print_cones(self):
        prev_x = self.cones[0][0]
        print_line = f"x={prev_x:.2f}: "
        for cone in self.cones:
            if cone[0] == prev_x:
                print_line += f"({cone[0]:.2f}, {cone[1]:.2f}) "
            else:
                print(print_line)  # 이전 x값의 모든 콘들을 출력
                prev_x = cone[0]
                print_line = f"x={prev_x:.2f}: ({cone[0]:.2f}, {cone[1]:.2f}) "

        print(print_line)  # 마지막 줄 출력

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        for cone in self.cones_shape:
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * 10), int(-y * 10)), 5)
        pygame.draw.line(self.screen, (0, 0, 0), (0, int(-8.0525*WINDOWWIDTH)), (WINDOWWIDTH, int(-8.0525*WINDOWWIDTH)), 5)
        for cone in self.cones:
            pygame.draw.circle(self.screen, (255, 165, 0), (int(cone[0]*WINDOWWIDTH), int(-cone[1]*WINDOWWIDTH)), 5)
        pygame.display.flip()

    def close(self):
        pygame.quit()
if __name__ == "__main__":
    carx, cary, caryaw = 20, -8, 0
    sight = 30
    env = Trajectory(carx, cary, caryaw, sight)
    print(env.print_cones())
    print(env.find_cones_closest_front(sight))
    env.plotting()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000 * 500)
