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
import math
from shapely.geometry import Polygon, Point, LineString
from SLALOM_cone2 import Road, Car, Cone
from shapely import affinity
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
    def __init__(self, host='127.0.0.1', port=10001, check=2, matlab_path='C:/CM_Projects/JX1_102/src_cm4sl', simul_path='pythonCtrl_SLALOM', use_carmaker = True):
        # Action과 State의 크기 및 형태를 정의.
        self.check = check
        self.use_carmaker = use_carmaker
        self.road_type = "SLALOM"
        self.road = Road()
        self.cone = Cone()
        env_action_num = 1
        sim_action_num = env_action_num + 1

        env_obs_num = 16
        sim_obs_num = 13
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

        if self.use_carmaker:
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

            self.test_num = 0

            self.traj_data = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj_SLALOM.csv").loc[:, ["traj_tx", "traj_ty"]].values


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

    def step(self, action1):
        action = np.append(action1, self.test_num)
        self.test_num += 1
        state_for_info = np.zeros(16)
        time = 0
        dev = np.array([0, 0])
        alHori = 0
        car_pos = np.array([2, -5.25, 0])
        car_dev = np.array([0, 0])
        car_steer = np.array([0, 0, 0])
        collision = 0
        car_v = 0
        car_roll = 0

        done = False

        # 최초 실행시
        if self.sim_initiated == False:
            self.sim_initiated = True

        # 에피소드의 첫 스텝
        if self.sim_started == False:
            self.status_queue.put("start")
            self.sim_started = True

        # Action 값 전송
        # CarMakerEnv -> CMcontrolNode -> tcpip_thread -> simulink tcp/ip block
        self.action_queue.put(action)

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
            # 튜플로 넘어온 값을 numpy array로 변환
            state = np.array(state) #어레이 변환
            state = state[1:]
            state_for_info = state
            time = state[0]
            car_pos = state[1:4] #x, y, yaw
            car_v = state[4] #1
            car_steer = state[5:8]
            dev = self.calculate_dev(car_pos[0], car_pos[1], car_pos[2])
            alHori = state[10]
            roll = state[11]
#            wheel_steer = state[12:]
            lookahead_sight = [2 * i for i in range(5)]
            lookahead_traj_abs = self.find_lookahead_traj(car_pos[0], car_pos[1], lookahead_sight)
            lookahead_traj_rel = self.to_relative_coordinates(car_pos[0], car_pos[1], car_pos[2], lookahead_traj_abs).flatten()

            lookahead_cones_abs = self.cone.cones_arr[self.road.cones_arr[:, 0] > car_pos[0]][:2]
            lookahead_cones_rel = self.to_relative_coordinates(car_pos[0], car_pos[1], car_pos[2], lookahead_cones_abs).flatten()

            state = np.concatenate((np.array([car_v, car_steer[0]]), lookahead_traj_rel, lookahead_cones_rel))


        # 리워드 계산
        reward_state = np.concatenate((dev, np.array([alHori]), car_pos))
        reward = self.getReward(reward_state, time)
        info_key = np.array(["time", "x", "y", "yaw", "carv", "ang", "vel", "acc", "devDist", "devAng", "alHori", "roll", "rl", "rr", "fl", "fr"])
        info = {key: value for key, value in zip(info_key, state_for_info)}

        return state, reward, done, info

    def find_lookahead_traj(self, x, y, distances):
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

        return result_points

    def to_relative_coordinates(self, x, y, yaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - x
            dy = point[1] - y

            rotated_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
            rotated_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def getReward(self, state, time):

        if state.any() == False:
            # 에피소드 종료시
            return 0.0

        dev_dist = abs(state[0])
        dev_ang = abs(state[1])
        alHori = abs(state[2])
        carx, cary, caryaw = state[3:]

        car = Car(carx, cary, caryaw)
        car_shape = car.shape_car(carx, cary, caryaw)
        if self.road.is_car_colliding_with_cones(car_shape):
            forbidden_reward = 10000
        else:
            forbidden_reward = 0

        #devDist, devAng에 따른 리워드
        reward_devDist = dev_dist * 1000
        reward_devAng = dev_ang * 5000

        e = - reward_devDist - reward_devAng - forbidden_reward

        if self.test_num % 300 == 0 and self.check == 0:
            print(f"Time: {time}, Reward : [ dist : {round(dev_dist,3)}] [ angle : {round(dev_ang, 3)}]")
        if forbidden_reward != 0:
            print("collision")

        return e

    def calculate_dev(self, carx, cary, caryaw):
        arr = np.array(self.traj_data)
        distances = np.sqrt(np.sum((arr - [carx, cary]) ** 2, axis=1))
        dist_index = np.argmin(distances)
        devDist = distances[dist_index]

        dx1 = arr[dist_index + 1][0] - arr[dist_index][0]
        dy1 = arr[dist_index + 1][1] - arr[dist_index][1]

        dx2 = arr[dist_index][0] - arr[dist_index - 1][0]
        dy2 = arr[dist_index][1] - arr[dist_index - 1][1]

        # 분모가 0이 될 수 있는 경우에 대한 예외처리
        if dx1 == 0:
            devAng1 = np.inf if dy1 > 0 else -np.inf
        else:
            devAng1 = dy1 / dx1

        if dx2 == 0:
            devAng2 = np.inf if dy2 > 0 else -np.inf
        else:
            devAng2 = dy2 / dx2

        devAng = - np.arctan((devAng1 + devAng2) / 2) - caryaw
        return np.array([devDist, devAng])

if __name__ == "__main__":
    # 환경 테스트
    env = CarMakerEnv(check=0, simul_path='test_Nomove')
    act_lst = []
    next_state_lst = []
    info_lst = []


    for i in range(1):
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
