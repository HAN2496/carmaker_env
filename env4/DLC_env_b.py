"""
강화학습에 사용할 수 있는 Gym Env 기반 카메이커 연동 클라스
cm_control.py에 구현된 기능을 이용
"""

import gym
from gym import spaces
import numpy as np
from cm_control import CMcontrolNode
import threading
from queue import Queue
import pandas as pd
import time
from DLC_env_low import CarMakerEnv as LowLevelCarMakerEnv
from stable_baselines3 import PPO, SAC
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, LineString
from DLC_cone import Road, Car
import pygame
from DLC_data2 import Data
from common_functions import *

# 카메이커 컨트롤 노드 구동을 위한 쓰레드
# CMcontrolNode 내의 sim_start에서 while loop로 통신을 처리하므로, 강화학습 프로세스와 분리를 위해 별도 쓰레드로 관리

def cm_thread(host, port, action_queue, state_queue, action_num, state_num, status_queue, matlab_path, simul_path):
    cm_env = CMcontrolNode(host=host, port=port, action_queue=action_queue, state_queue=state_queue, action_num=action_num, state_num=state_num, matlab_path=matlab_path, simul_path=simul_path)

    while True:
        # 강화학습에서 카메이커 시뮬레이션 상태를 결정
        status = status_queue.get()
        if status == "start":
            # 시뮬레이션 시작
            # TCP/IP 로드 -> 카메이커 시뮬레이션 시작 -> 강화학습과 데이터를 주고 받는 loop
            cm_env.start_sim()
        elif status == "stop":
            # 시뮬레이션 종료
            cm_env.stop_sim()
        elif status == "finish":
            # 프로세스 종료
            cm_env.kill()
            break
        else:
            time.sleep(1)

class CarMakerEnvB(gym.Env):
    def __init__(self, host='127.0.0.1', port=10001, check=2, matlab_path='C:/CM_Projects/JX1_102/src_cm4sl', simul_path='pythonCtrl_DLC'):
        # Action과 State의 크기 및 형태를 정의.
        self.check = check
        self.road_type = "DLC"
        self.data = Data()
        
        #env에서는 1개의 action, simulink는 connect를 위해 1개가 추가됨
        env_action_num = 1
        sim_action_num = env_action_num + 1

        # Env의 observation 개수와 simulink observation 개수
        env_obs_num = 20
        sim_obs_num = 17

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        # 카메이커 연동 쓰레드와의 데이터 통신을 위한 큐
        self.status_queue = Queue()
        self.action_queue = Queue()
        self.state_queue = Queue()

        # 중복 명령 방지를 위한 변수
        self.sim_initiated = False
        self.sim_started = False

        # 각 Env마다 1개의 카메이커 연동 쓰레드를 사용
        self.cm_thread = threading.Thread(target=cm_thread, daemon=False, args=(host,port,self.action_queue, self.state_queue, sim_action_num, sim_obs_num, self.status_queue, matlab_path, simul_path))
        self.cm_thread.start()

        self.car = Car()
        self.road = Road()
        self.test_num = 0
        self.traj_data = np.array([[3, -10], [15, -10]])
        self.car_data = np.array([2, -10, 0, 13.8889, 0])
        self.traj_data = self.make_trajectory(self.traj_data[0], self.traj_data[1])
        self.traj_point = self.find_nearest_point(2, -10, [3*i for i in range(5)])
        low_level_env = LowLevelCarMakerEnv(use_carmaker=False)
#        self.low_level_model = SAC.load(f"models/{self.road_type}/512399_best_model.pkl", env=low_level_env)
        self.low_level_model = SAC.load(f"1st_best_model.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()
        self.before_traj_point = np.array([15, -10])

        if self.check == 0:
            pygame.init()
            self.screen = pygame.display.set_mode((self.road.road_length * 10, - self.road.road_width * 10))
            pygame.display.set_caption("B level Environment")

    def __del__(self):
        self.cm_thread.join()

    def _initial_state(self):
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.traj_data = np.array([[3, -10], [15, -10]])
        self.traj_data = self.make_trajectory(self.car_data[0], self.car_data[1])
        if self.check == 0:
            self.render()

        # 초기화 코드
        if self.sim_initiated == True:
            # 한번의 시뮬레이션도 실행하지 않은 상태에서는 stop 명령을 줄 필요가 없음
            self.status_queue.put("stop")
            self.action_queue.queue.clear()
            self.state_queue.queue.clear()
        self.sim_started = False

        return self._initial_state()


    def step(self, action):
        """
        Simulink의 tcpiprcv와 tcpipsend를 연결
        기존 action : steering vel -> scalar
        Policy b action : traj points -> array(list)
        low_level_obs에 cary, carv 들어가고, car_dev랑 lookahead는 action(신규)에서 가져온 trj 정보로
        """
        self.test_num += 1
        done = False

        time = 0
        carx, cary, caryaw = np.array([0, 0, 0])
        car_v = 0
        car_steer = np.array([0, 0, 0])
        car_dev = np.array([0, 0])
        car_alHori = 0
        car_roll = 0
        new_traj_point = self.make_traj_point(self.car_data[0], self.car_data[1], 0)
        sight = np.array([3 * i for i in range(5)])

        traj_lowlevel_abs = self.find_nearest_point(self.car_data[0], self.car_data[1], sight)
        traj_lowlevel_rel = self.to_relative_coordinates(self.car_data[0], self.car_data[1], self.car_data[2], traj_lowlevel_abs).flatten()
        self.low_level_obs = np.concatenate((np.array([self.car_data[3], self.car_data[4]]), traj_lowlevel_rel))
        steering_changes = self.low_level_model.predict(self.low_level_obs)
        action_to_sim = np.append(steering_changes[0], self.test_num)

        # 최초 실행시
        if self.sim_initiated == False:
            self.sim_initiated = True

        # 에피소드의 첫 스텝
        if self.sim_started == False:
            self.status_queue.put("start")
            self.sim_started = True

        # Action 값 전송 / State 값 수신
        self.action_queue.put(action_to_sim)
        state = self.state_queue.get()

        if state == False:
            # 시뮬레이션 종료
            # 혹은 여기 (끝날때마다)
            # CMcontrolNode에서 TCP/IP 통신이 종료되면(시뮬레이션이 끝날 경우) False 값을 리턴하도록 정의됨
            state = self._initial_state()
            done = True

        else:
            blevel_action = action[0]
            # 튜플로 넘어온 값을 numpy array로 변환
            state = np.array(state) #어레이 변환
            state = state[1:] #connect 제거
            time = state[0] # Time
            carx, cary, caryaw, carv = state[1:5]
            self.car.carx, self.car.cary, self.car.caryaw, self.car.carv = carx, cary, caryaw, carv
            car_steer = state[5:8] #Car.Steer.(Ang, Vel, Acc)
            car_dev = state[8:10] #Car.DevDist, Car.DevAng
            car_alHori = state[10] #alHori
            car_roll = state[11]
            wheel_steer = state[12:]

            new_traj_point = self.make_traj_point(carx, cary, blevel_action)
            self.traj_data = self.make_trajectory(carx, cary, blevel_action)
            traj_point = self.to_relative_coordinates(carx, cary, caryaw, np.vstack((self.before_traj_point, new_traj_point))).flatten()
            traj_abs = self.find_nearest_point(carx, cary, sight)
            self.traj_point = traj_abs
            traj_rel = self.to_relative_coordinates(carx, cary, caryaw, traj_abs).flatten()
            car_dev = self.calculate_dev(carx, cary, caryaw)
            cones_abs = self.road.cones_arr[self.road.cones_arr[:, 0] > carx][:3]
            cones_rel = self.to_relative_coordinates(carx, cary, caryaw, cones_abs).flatten()

            cones_for_lowlevel = self.road.cones_arr[self.road.cones_arr[:, 0] > carx][:2]
            cones_rel_for_lowlevel = self.to_relative_coordinates(carx, cary, caryaw, cones_for_lowlevel).flatten()
            self.car_data = np.array([carx, cary, caryaw, carv, car_steer[0]])

            state = np.concatenate((traj_point, traj_rel, cones_rel)) # <- Policy B의 state
            print(f"shae: {np.shape(state)}")

        # 리워드 계산
        reward = self.getReward(new_traj_point, time)
        info = {"Time" : time, "Steer.Ang" : car_steer[0], "Steer.Vel" : car_steer[1], "Steer.Acc" : car_steer[2], "carx" : carx, "cary" : cary,
                "caryaw" : caryaw, "carv" : car_v, "alHori" : car_alHori, "Roll": car_roll}

        self.before_traj_point = new_traj_point

        if self.test_num % 300 == 0:
            self.print_result(time, reward, car_dev)

        if self.check == 0:
            self.render()
        return state, reward, done, info

    def make_traj_point(self, carx, cary, action):
        new_traj_point = np.array([carx + 12, cary + action * 3])
        return new_traj_point

    def make_trajectory(self, carx, cary, action=0):
        arr = self.traj_data.copy()

        if action != 0:
            new_traj_point = self.make_traj_point(carx, cary, action)
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
            x_diff = np.abs(self.traj_data[:, 0] - (x0 + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(self.traj_data[nearest_idx])
        return points

    def calculate_dev(self, carx, cary, caryaw):
        arr = np.array(self.traj_data)
        distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]

        dx = arr[dist_index][0] - arr[dist_index - 1][0]
        dy = arr[dist_index][1] - arr[dist_index - 1][1]

        # 분모가 0이 될 수 있는 경우에 대한 예외처리
        if dx == 0:
            devAng = np.inf if dy > 0 else -np.inf
        else:
            devAng = dy / dx

        return np.array([devDist, devAng])

    def to_relative_coordinates(self, carx, cary, caryaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - carx
            dy = point[1] - cary

            rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
            rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def getReward(self, new_traj_point, time):
        car = Car()
        car_shape = car.shape_car(self.car_data[0], self.car_data[1], self.car_data[2])
        forbidden_reward, cones_reward, car_reward, ang_reward = 0, 0, 0, 0
        traj_point = Point(new_traj_point[0], new_traj_point[1])
        if self.road.forbbiden_area1.intersects(traj_point) or self.road.forbbiden_area2.intersects(traj_point):
            forbidden_reward = -10000
        if self.road.cones_boundary.intersects(traj_point):
            cones_reward = +100
        if self.road.is_car_in_forbidden_area(car_shape):
            car_reward = -10000
        traj_reward = - np.linalg.norm((new_traj_point - self.before_traj_point)) * 1000

        e = forbidden_reward + cones_reward + car_reward + ang_reward + traj_reward
        return e

    def save_data_for_lowlevel(self, indexs, values):
        datas = {}
        for idx, index in enumerate(indexs):
            datas[index] = values[i]
        return datas

    def print_result(self, time, reward, car_dev):
        print("-" * 50)
        print(
            f"[Time: {round(time, 2)}] [Reward: {round(reward, 2)}] [Car dev: {round(car_dev[0], 2), round(car_dev[1], 2)}]")
        print("[Trajectory: ]")
        for point in self.traj_point:
            print(f" [{point[0]:.2f}, {point[1]:.2f}]")

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

if __name__ == "__main__":
    # 환경 테스트
    env = CarMakerEnvB(check=0)
    act_lst = []
    next_state_lst = []
    info_lst = []


    for i in range(3):
        # 환경 초기화
        state = env.reset()

        # 에피소드 실행
        done = False
        while not done:
            action = env.action_space.sample()  # 랜덤 액션 선택
            if i==0:
                act_lst.append(action)
                df = pd.DataFrame(data=act_lst)
            next_state, reward, done, info = env.step(action)

            if i==0:
                next_state_lst.append(next_state)
                info_lst.append(info)

            if done == True:
                print("Episode Finished.")
                df.to_csv('env_action_check.csv')

                df2 = pd.DataFrame(data=next_state_lst)
                df2.to_csv('env_state_check.csv', index=False)

                df3 = pd.DataFrame(data=info_lst)
                df3.to_csv('env_info_check.csv', index=False)

                break
