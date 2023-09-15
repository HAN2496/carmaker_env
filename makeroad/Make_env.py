import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pygame

class MakeRoadEnv(gym.Env):
    def __init__(self):
        super(MakeRoadEnv, self).__init__()
        self.time = 0

        self.car_width = 1.568
        self.car_length = 4.3
        self.car_angle = 0
        self.car_v = 13.8889 #50 kph
        self.car_pos = np.array([0, -8.25])

        self.cone_pos = self.create_DLC_cone()

        self.road_length = 161
        self.road_width = 30

        env_action_num = 1
        env_obs_num = 214
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.road_length * 10, self.road_width * 10))  # Multiplied to scale up for visibility
        pygame.display.set_caption("Car Road Environment")

    def _initial_state(self):
        self.time = 0
        self.car_pos = np.array([0, -8.25])
        self.car_angle = 0
        return np.zeros(self.observation_space.shape)

    def reset(self):
        return self._initial_state()

    def close(self):
        pygame.quit()

    def step(self, action):
        done = False
        self.render()

        steering_changes = action[0]
        self.car_angle += steering_changes
        self.car_pos[0] += np.cos(np.radians(self.car_angle)) * self.car_v * 0.01
        self.car_pos[1] += np.sin(np.radians(self.car_angle)) * self.car_v * 0.01
        state = np.concatenate((self.car_pos, self.cone_pos.flatten()))
        reward = self.getReward(state)
        info = {}

        if self.car_pos[1] >= self.road_length:
            done = True

        self.time += 0.01

        return state, reward, done, info

    def getReward(self, state):
        for cone in self.cone_pos:
            if self.check_collision(cone):
                return -100
        return 0

    def check_collision(self, cone):
        # Basic rectangle-point collision check
        car_x_min = self.car_pos[0] - self.car_width / 2
        car_x_max = self.car_pos[0] + self.car_width / 2
        car_y_min = self.car_pos[1]
        car_y_max = self.car_pos[1] + self.car_length

        return car_x_min <= cone[0] <= car_x_max and car_y_min <= cone[1] <= car_y_max


    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fill background (road)
        self.screen.fill((128, 128, 128))  # Gray for road

        # Draw cones
        for cone in self.cone_pos:
            pygame.draw.circle(self.screen, (255, 140, 0), (int(cone[0] * 10), int(self.road_length*10 - cone[1] * 10)), 5)

        # Calculate the car's corner positions based on its angle and current position
        cos_angle = np.cos(np.radians(self.car_angle))
        sin_angle = np.sin(np.radians(self.car_angle))

        half_width = self.car_width / 2
        half_length = self.car_length / 2

        corners = [
            (self.car_pos[0] + half_length * cos_angle - half_width * sin_angle,
             self.car_pos[1] + half_length * sin_angle + half_width * cos_angle),
            (self.car_pos[0] + half_length * cos_angle + half_width * sin_angle,
             self.car_pos[1] + half_length * sin_angle - half_width * cos_angle),
            (self.car_pos[0] - half_length * cos_angle + half_width * sin_angle,
             self.car_pos[1] - half_length * sin_angle - half_width * cos_angle),
            (self.car_pos[0] - half_length * cos_angle - half_width * sin_angle,
             self.car_pos[1] - half_length * sin_angle + half_width * cos_angle)
        ]
        car_rect = pygame.Rect(int(self.car_pos[0] * 10), int(self.road_length*10 - (self.car_pos[1] + self.car_length) * 10), int(self.car_length * 10), int(self.car_width * 10))
        pygame.draw.rect(self.screen, (0, 0, 255), car_rect)  # Blue for the car

#        pygame.draw.polygon(self.screen, (0, 0, 255), [(x * 10, y * 10) for x, y in corners])  # Multiplied position by 10 to match the scale

        pygame.display.flip()
        pygame.time.wait(50)



    def create_cone(self, sections):
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
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2.52, 'num': 5, 'y_offset': -4.8315}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 5.981, 'num': 4, 'y_offset': -6.562},
            {'start': 99, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 111, 'gap': 5, 'cone_dist': 3, 'num': 20, 'y_offset': -8.0525}
        ]

        return self.create_cone(sections)

if __name__ == "main":
    env = MakeRoadEnv()
    model = SAC.load("model.pkl", env=env)
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
