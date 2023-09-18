import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import pygame
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt

def plot(road, car):
    plt.figure(figsize=(10, 5))

    # Plot forbidden areas
    plt.plot(*road.forbbiden_area1.exterior.xy, label="Forbidden Area 1", color='red')
    plt.plot(*road.forbbiden_area2.exterior.xy, label="Forbidden Area 2", color='blue')
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
    plt.gca().invert_yaxis()
    plt.title('Car, Forbidden Areas and Cones')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

class Road:
    def __init__(self):
        self.road_length = 161
        self.road_width = -15
        self._forbidden_area()
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()

    def _forbidden_area(self):
        vertices1 = [
            (0, -6.442),
            (62, -6.442),
            (62, -3.221),
            (99, -3.221),
            (99, -6.442),
            (161, -6.442),
            (161, 0),
            (0, 0),
            (0, -6.442)
        ]
        vertices2 = [
            (0, -9.663),
            (75.5, -9.663),
            (75.5, -6.442),
            (86.5, -6.442),
            (86.5, -9.663),
            (161, -9.663),
            (161, -12.884),
            (0, -12.884),
            (0, -9.663)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
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
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2.52, 'num': 5, 'y_offset': -4.8315}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 5.981, 'num': 4, 'y_offset': -6.562},
            {'start': 99, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 111, 'gap': 5, 'cone_dist': 3, 'num': 20, 'y_offset': -8.0525}
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

class Car:
    def __init__(self):
        self.length = 4.3
        self.width = 1.568
        self.carx = 5
        self.cary = -8.0525
        self.caryaw = 0
        self.carv = 13.8889

    def reset_car(self):
        self.carx = 5
        self.cary = -8.0525
        self.caryaw = 0

    def move_car(self, angle):
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

class MakeRoadEnv(gym.Env):
    def __init__(self):
        super(MakeRoadEnv, self).__init__()
        self.reset_num = 0
        self.time = 0
        self.test_num = 0

        self.car = Car()
        self.road = Road()

        env_action_num = 1
        env_obs_num = 215
        self.action_space = spaces.Box(low=-1, high=1, shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.road.road_length * 10, - self.road.road_width * 10))
        pygame.display.set_caption("Car Road Environment")

    def _initial_state(self):
        self.time = 0
        self.car.reset_car()
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.reset_num += 1
        print(f'reset : {self.reset_num}')
        return self._initial_state()

    def close(self):
        pygame.quit()

    def step(self, action):
        done = False
        self.test_num += 1
        self.render()

        steering_changes = action
        self.car.move_car(steering_changes[0])
        state = np.concatenate((np.array([self.car.carx, self.car.cary, self.car.caryaw]), self.road.cones_arr.flatten()))

        if self.test_num % 100 == 0:
            print(f"[Time: {round(self.time, 2)}, Action : {action}, Ang : {round(self.car.caryaw, 2)}, Carx: {round(self.car.carx, 2)}, Cary: {round(self.car.cary, 2)}")

        if self.road.is_car_in_road(self.car) == 1:
            done = True
        reward = self.getReward(state)
        info = {}

        self.time += 0.01

        return state, reward, done, info

    def getReward(self, state):
        reward = 0
        if self.road.is_car_in_forbidden_area(self.car):
            reward = -2000
        elif self.road.is_car_colliding_with_cones(self.car):
            reward = -1000
        if self.test_num % 100 == 0:
            print(f"Reward : {reward}")
        return reward

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((128, 128, 128))

        for cone in self.road.cones_shape:
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * 10), int(-y * 10)), 5)

        car_shape = self.car.shape_car(self.car.carx, self.car.cary, self.car.caryaw)
        car_color = (255, 0, 0)
        pygame.draw.polygon(self.screen, car_color, [(int(x*10), int(-y*10)) for x, y in car_shape.exterior.coords])

        pygame.display.flip()

if __name__ == "__main__":
    env = MakeRoadEnv()
    plot(env.road, env.car)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000 * 500)
    """
    env = MakeRoadEnv()
    model = SAC.load("model.pkl", env=env)
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
    """
