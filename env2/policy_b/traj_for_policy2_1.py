import numpy as np
import gym
import pygame
from gym import spaces
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DQN
class Road:
    def __init__(self):
        self.road_length = 161
        self.road_width = 12.884

        #콘 배열과 shape 세팅
        self.cones_arr, self.cones_shape = self.create_cone()
        
        #로드 바운더리, 금지영역 세팅
        self.road_boundary = self.create_road_boundary()
        self.forbidden_area1, self.forbidden_area2 = self.create_forbidden_area()



    def create_cone(self):
        sections = self.create_DLC_sections()
        cones_arr = []
        cones_shape = []
        for section in sections:
            for i in range(section['num']):  # Each section has 5 pairs
                x_base = section['start'] + section['gap'] * i
                y1 = section['y_offset'] - section['cone_dist'] / 2
                y2 = section['y_offset'] + section['cone_dist'] / 2
                cones_arr.extend([[x_base, y1], [x_base, y2]])
                cones_shape.extend([Point(x_base, y1).buffer(0.2), Point(x_base, y2).buffer(0.2)])

        return np.array(cones_arr), np.array(cones_shape)
    def create_DLC_sections(self):
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

    def create_road_boundary(self):
        return Polygon([
            (-1, 0), (self.road_length + 1, 0), (self.road_length + 1, -self.road_width), (-1, -self.road_width)])

    def create_forbidden_area(self):
        vertices1 = [
            (0, -6.442), (62, -6.442), (62, -3.221), (99, -3.221), (99, -6.442),
            (161, -6.442), (161, 0), (0, 0), (0, -6.442)
        ]
        vertices2 = [
            (0, -9.663), (75.5, -9.663), (75.5, -6.442), (86.5, -6.442),
            (86.5, -9.663), (161, -9.663), (161, -12.884), (0, -12.884), (0, -9.663)
        ]
        forbidden_area1 = Polygon(vertices1)
        forbidden_area2 = Polygon(vertices2)
        self.forbidden_line1 = LineString(vertices1[:])
        self.forbidden_line2 = LineString(vertices2[:])
        return forbidden_area1, forbidden_area2

    def plot_road(self, show=True):
        roadx, roady = self.road_boundary.exterior.xy
        plt.fill(roadx, roady, color=(128/255, 128/255, 128/255), alpha=0.2, label='Road')
        area1x, area1y = self.forbidden_area1.exterior.xy
        plt.fill(area1x, area1y, alpha=0.3)
        plt.plot(area1x, area1y, label='forbidden area')
        area2x, area2y = self.forbidden_area2.exterior.xy
        plt.fill(area2x, area2y, alpha=0.3)
        plt.plot(area2x, area2y)

        plt.scatter(self.cones_arr[:,0], self.cones_arr[:, 1], label='Cone')
        if show:
            plt.legend()
            plt.show()

class Trajectory(gym.Env):
    def __init__(self):
        super(Trajectory, self).__init__()
        self.test_num = 0
        self.road = Road()
        self.traj_arr = self.init_trajectory_arr()
        self.traj_line = self.make_trajectory_line()

        actionx, actiony = 161, 3
        env_obs_num = 322
        self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([160, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.road.road_length * 10, self.road.road_width * 10))
        pygame.display.set_caption("Trajectory")
    def init_trajectory_arr(self):
        return np.array([[i, -8.0525] for i in range(161)])
    def make_trajectory_line(self):
        return LineString(self.traj_arr)

    def _initial_state(self):
        self.traj_arr = self.init_trajectory_arr()
        self.traj_line = self.make_trajectory_line()
        return np.zeros(self.observation_space.shape)
    def reset(self):
        return self._initial_state()
    def move_traj_pos(self, x, y):
        self.traj_arr[int(x)][1] += y
        self.traj_line = self.make_trajectory_line()

    def step(self, action):
        done = False
        self.test_num += 1
        self.render()

        actionx = action[0]
        actiony = action[1]

        self.move_traj_pos(actionx, actiony)
        state = self.traj_arr.flatten()

        reward = self.getReward()
        for traj_point in self.traj_arr:
            if not self.road.road_boundary.contains(Point(traj_point)):
                done = True
        info = {}

        if self.test_num % 300 == 0:
            print(f"[Reward: {reward}]")

        return state, reward, done, info

    def getReward(self):
        if self.road.forbidden_area1.intersects(self.traj_line):
            forbidden_reward = -1000
        else:
            forbidden_reward = 0
        e = + forbidden_reward
        return e
    def plot(self):
        self.road.plot_road(show=False)
        plt.scatter(self.traj_arr[:, 0], self.traj_arr[:, 1], color='red', label='Trajectory')
        x, y = self.traj_line.xy
        plt.plot(x, y, label='Trajectory Line')
        plt.legend()
        plt.show()

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        for cone in self.road.cones_arr:
            x, y = cone[0], cone[1]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * 10), int(-y * 10)), 5)

        for traj in self.traj_arr:
            x, y = traj[0], traj[1]
            pygame.draw.circle(self.screen, (255, 0, 0), (int(x * 10), int(-y * 10)), 5)

        for i in range(len(self.traj_arr) - 1):
            x1, y1 = self.traj_arr[i]
            x2, y2 = self.traj_arr[i + 1]
            pygame.draw.line(self.screen, (255, 0, 0), (int(x1 * 10), int(-y1 * 10)), (int(x2 * 10), int(-y2 * 10)), 5)

        pygame.display.flip()
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = Trajectory()
    env.move_traj_pos(10, 2)
    #env.plot()
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000 * 500)