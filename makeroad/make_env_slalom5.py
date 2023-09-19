import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import pygame
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
import math
import time
#from utils import scale_image, blit_rotate_center

XMULTIPLE, YMULTIPLE = 2, 10

def plot(road, car):

    plt.plot(*road.road_boundary.exterior.xy, label="Forbidden Area 1", color='red')

    cones_x = road.cones_arr[:, 0]
    cones_y = road.cones_arr[:, 1]
    plt.scatter(cones_x, cones_y, s=10, color='orange', label="Cones")

    car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
    plt.plot(*car_shape.exterior.xy, color='blue', label="Car")
    plt.scatter(car.carx, car.cary, color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Car, Forbidden Areas and Cones')
    plt.legend()
    plt.grid(True)
    plt.ylim([-15, 5])
    plt.axis('equal')
    plt.show()

class Road:
    def __init__(self):
        self.road_length = 500
        self.road_width = -15
        self.road_boundary = self._create_road_boundary()
        self.cones_arr, self.cones_shape = self._create_SLALOM_cone()
        self.middle_arr, self.middle_shape = self._create_SLALOM_middle()

    def _create_road_boundary(self):
       polygon = Polygon(
            [(0, 5), (self.road_length, 5), (self.road_length, self.road_width), (0, self.road_width)])
       return polygon


    def _create_SLALOM_cone(self):
        first_cone_arr = [[i, -5.25] for i in range(100, 400, 30)]
        second_cone_arr = [[i, -5.25] for i in range(600, 800, 30)]
        cones_arr = np.array(first_cone_arr + second_cone_arr)
        cones_shape = np.array([Point(cx, cy).buffer(0.2) for cx, cy in cones_arr])
        return cones_arr, cones_shape

    def _create_SLALOM_middle(self):
        first_cone_arr = [[i, -5.25] for i in range(115, 400, 30)]
        second_cone_arr = [[i, -5.25] for i in range(615, 800, 30)]
        cones_arr = np.array(first_cone_arr + second_cone_arr)
        cones_shape = np.array([Point(cx, cy).buffer(0.2) for cx, cy in cones_arr])
        return cones_arr, cones_shape

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
        self.carx = 2.5
        self.cary = -5.25
        self.caryaw = 0
        self.carv = 13.8889

    def reset_car(self):
        self.carx = 5
        self.cary = -5.25
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
        env_obs_num = 24
        self.action_space = spaces.Box(low=-1, high=1, shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.road.road_length * XMULTIPLE, - self.road.road_width * YMULTIPLE))
        pygame.display.set_caption("Car Road Environment - SLALOM 3")

    def _initial_state(self):
        self.time = 0
        self.car.reset_car()
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.reset_num += 1
#        print(f'reset : {self.reset_num}')
        return self._initial_state()

    def close(self):
        pygame.quit()

    def step(self, action):
        done = False
        self.test_num += 1
        self.render()

        steering_changes = action
        self.car.move_car(steering_changes[0])
        cones_sight = self.cone_in_sight(self.car.carx, 5)
        middles_sight = self.middle_in_sight(self.car.carx, 5)
        cones_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, cones_sight)
        middles_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, middles_sight)
        state = np.concatenate((np.array([self.car.carx, self.car.cary, self.car.caryaw, self.car.carv]), cones_rel.flatten(), middles_rel.flatten()))


        if self.road.is_car_in_road(self.car) == 1:
            done = True
        reward = self.getReward()
        info = {"carx": self.car.carx, "cary": self.car.cary, "caryaw": self.car.caryaw}

        self.time += 0.01

        return state, reward, done, info

    def switch_state(self):
        if self.car.carx <= 70 or self.car.cary >= 130:
            pass
        else:

    def getReward(self):
        reward = 0

        # 가장 가까운 콘과의 거리 찾기
        min_distance = 0
        for cone in self.road.cones_arr:
            distance_to_cone = np.linalg.norm(np.array([self.car.carx, self.car.cary]) - np.array(cone))
            min_distance = min(min_distance, distance_to_cone)

        # 거리가 증가할수록 더 큰 벌점을 받도록 설정
        if min_distance >= 2.65:
            reward -= 0
        else:
            reward -= 10 * math.exp(min_distance - 2.65)

        mid_min_distance = 0
        for mid in self.road.middle_arr:
            distance_to_mid = np.linalg.norm(np.array([self.car.carx, self.car.cary]) - np.array(mid))
            mid_min_distance = min(min_distance, distance_to_mid)

        if mid_min_distance >= 5:
            reward -=0
        else:
            reward -= 400 * mid_min_distance

        # 콘과 충돌 시 큰 벌점 부여
        if self.road.is_car_colliding_with_cones(self.car):
            reward -= 2000

        y_reward = abs(self.car.cary + 5.25) * 200
        yaw_reward = abs(self.car.caryaw) * 400
        reward -= y_reward
        reward -= yaw_reward
        if self.test_num % 200 == 0:
            print(f"[Time: {round(self.time, 2)}, Reward : {reward}, Ang : {round(self.car.caryaw, 4)}, Carx: {round(self.car.carx, 2)}, Cary: {round(self.car.cary, 2)}")

        return reward

    def cone_in_sight(self, carx, sight):
        return np.array([cone for cone in self.road.cones_arr if carx - 2.15 <= cone[0]][:sight])

    def middle_in_sight(self, carx, sight):
        return np.array([cone for cone in self.road.cones_arr if carx - 2.15 <= cone[0]][:sight])

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
            pygame.draw.circle(self.screen, (255, 140, 0), (int(x * XMULTIPLE), int(-y * YMULTIPLE)), 5)

        # Create a Surface for the car.
        car_color = (255, 0, 0)

        half_length = self.car.length * XMULTIPLE / 2.0
        half_width = self.car.width * YMULTIPLE / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        # Rotate the car corners
        rotated_corners = []
        for x, y in corners:
            x_rot = x * np.cos(-self.car.caryaw) - y * np.sin(-self.car.caryaw) + self.car.carx * XMULTIPLE
            y_rot = x * np.sin(-self.car.caryaw) + y * np.cos(-self.car.caryaw) - self.car.cary * YMULTIPLE
            rotated_corners.append((x_rot, y_rot))

        # Draw the car on the main screen using the rotated corners
        pygame.draw.polygon(self.screen, car_color, rotated_corners)

        pygame.display.flip()


if __name__ == "__main__":
    env = MakeRoadEnv()
    plot(env.road, env.car)
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000 * 500)
    model.save("Model3.pkl")
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
