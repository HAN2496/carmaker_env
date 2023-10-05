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
from low_env_DLC import CarMakerEnv as LowLevelCarMakerEnv
from stable_baselines3 import PPO, SAC
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, LineString

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

class CarMakerEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=10001, check=2, matlab_path='C:/CM_Projects/PTC0910/src_cm4sl', simul_path='pythonCtrl_DLC'):
        # Action과 State의 크기 및 형태를 정의.
        self.check = check
        self.road_type = "DLC"
        
        #env에서는 1개의 action, simulink는 connect를 위해 1개가 추가됨
        env_action_num = 1
        sim_action_num = env_action_num + 1

        # Env의 observation 개수와 simulink observation 개수
        env_obs_num = 13
        sim_obs_num = 13

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

        self.road = Road()
        self.test_num = 0
        self.traj_data = np.array([[3, -8.0525], [15, -8.0525]])
        self.traj_data = self.make_trajectory()
        self.traj_point = self.find_nearest_point(2, -8.0525, [3*i for i in range(5)])
        self.car_before = np.array([2, -8.0525, 0, 13.8889])
        low_level_env = LowLevelCarMakerEnv(use_carmaker=False)
        self.low_level_model = SAC.load(f"model_forcheck/{self.road_type}/512399_best_model.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()

    def __del__(self):
        self.cm_thread.join()

    def _initial_state(self):
        return np.zeros(self.observation_space.shape)

    def reset(self):
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
        car_pos = np.array([0, 0, 0])
        car_v = 0
        car_steer = np.array([0, 0, 0])
        car_dev = np.array([0, 0])
        car_alHori = 0
        car_roll = 0
        sight = np.array([3 * i for i in range(5)])

        traj_lowlevel_abs = self.find_nearest_point(self.car_before[0], self.car_before[1], sight)
        traj_lowlevel_rel = self.to_relative_coordinates(self.car_before[1], self.car_before[1], self.car_before[2], traj_lowlevel_abs).flatten()
        self.low_level_obs = np.concatenate((np.array([self.car_before[3]]), traj_lowlevel_rel))
        steering_changes = self.low_level_model.predict(self.low_level_obs)
        action_to_sim = np.append(steering_changes, self.test_num)

        # 최초 실행시
        if self.sim_initiated == False:
            self.sim_initiated = True

        # 에피소드의 첫 스텝
        if self.sim_started == False:
            self.status_queue.put("start")
            self.sim_started = True

        # Action 값 전송
        # CarMakerEnv -> CMcontrolNode -> tcpip_thread -> simulink tcp/ip block
        self.action_queue.put(action_to_sim)

        # State 값 수신
        # simulink tcp/ip block -> tcpip_thread -> CMcontrolNode -> CarMakerEnv
        state = self.state_queue.get()


        if state == False:
            # 시뮬레이션 종료
            # 혹은 여기 (끝날때마다)
            # CMcontrolNode에서 TCP/IP 통신이 종료되면(시뮬레이션이 끝날 경우) False 값을 리턴하도록 정의됨
            state = self._initial_state()
            done = True

        else:
            new_traj_point = self.make_traj_point(action[0])
            self.traj_data = self.make_trajectory(action[0])
            # 튜플로 넘어온 값을 numpy array로 변환
            state = np.array(state) #어레이 변환
            state = state[1:] #connect 제거
            self.car_before = state[1:5]
            time = state[0] # Time
            carx, cary, caryaw = state[1:4]
            car_v = state[4] #Car.v
            car_steer = state[5:8] #Car.Steer.(Ang, Vel, Acc)
            car_dev = state[8:10] #Car.DevDist, Car.DevAng
            car_alHori = state[10] #alHori
            car_roll = state[11]
            new_traj_point = self.make_traj_point(action[0])
            self.traj_data = self.make_trajectory(action[0])
            traj_abs = self.find_nearest_point(carx, cary, sight)
            self.traj_point = traj_abs
            traj_rel = self.to_relative_coordinates(carx, cary, caryaw, traj_abs).flatten()
            car_dev = self.calculate_dev()
            cones_state = self.road.cones_arr[self.road.cones_arr[:, 0] > self.car.carx][:5]
            cones_rel = self.to_relative_coordinates(self.car.carx, self.car.cary, self.car.caryaw, cones_state).flatten()
            state = np.concatenate((traj_rel, cones_rel)) # <- Policy B의 state
        # 리워드 계산
        reward_state = np.concatenate((car_dev, np.array([car_alHori]), np.array([car_pos[0]])))
        reward = self.getReward(reward_state, time)
        info = {"Time" : time, "Steer.Ang" : car_steer[0], "Steer.Vel" : car_steer[1], "Steer.Acc" : car_steer[2], "carx" : car_pos[0], "cary" : car_pos[1],
                "caryaw" : car_pos[2], "carv" : car_v, "alHori" : car_alHori, "Roll": car_roll}
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

    def calculate_dev(self, car_pos):
        f = interp1d(self.traj_data[:, 0], self.traj_data[:, 1])
        xnew = np.arange(self.traj_data[0][0], self.traj_data[-1][0], 0.01)
        ynew = f(xnew)
        arr = np.array(list(zip(xnew, ynew)))
        distances = np.sqrt(np.sum((arr - [car_pos[0], car_pos[1]]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]
        if arr[dist_index][0] - arr[dist_index - 1][0] == 0:
            devAng2 = np.arctan(np.inf)
        else:
            devAng2 = np.arctan((arr[dist_index][1] - arr[dist_index - 1][1]) / (arr[dist_index][0] - arr[dist_index - 1][0]))
        devAng = - devAng2 - car_pos[2]
        return devDist, devAng

    def to_relative_coordinates(self, carx, cary, caryaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - carx
            dy = point[1] - cary

            rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
            rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)
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



if __name__ == "__main__":
    # 환경 테스트
    env = CarMakerEnv(check=0)
    act_lst = []
    next_state_lst = []
    info_lst = []


    for i in range(2):
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
