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
from SLALOM_cone2 import Road, Car, Cone
import pygame
from SLALOM_data2 import Data

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
    def __init__(self, host='127.0.0.1', port=10001, check=2, matlab_path='C:/CM_Projects/JX1_102/src_cm4sl', simul_path='pythonCtrl_SLALOM'):
        # Action과 State의 크기 및 형태를 정의.
        self.check = check
        self.road_type = "SLALOM"
        self.test_num = 0
        self.data = Data(check=check)
        self.road = Road()
        self.cone = Cone()

        #env에서는 1개의 action, simulink는 connect를 위해 1개가 추가됨
        env_action_num = 1
        sim_action_num = env_action_num + 1

        # Env의 observation 개수와 simulink observation 개수
        env_obs_num = self.data.state_size()
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

        low_level_env = LowLevelCarMakerEnv(use_carmaker=False)
        self.low_level_model = SAC.load(f"best_model/SLALOM_env1_best_model_compatible.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()

    def __del__(self):
        self.cm_thread.join()

    def _initial_state(self):
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.data._init()
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

        reward_argument = self.data._init_reward_argument()
        info = self.data._init_info()

        traj_lowlevel_abs = self.data.find_traj_points(self.data.carx)
        traj_lowlevel_rel = self.data.to_relative_coordinates(traj_lowlevel_abs).flatten()
        self.low_level_obs = np.concatenate((np.array([self.data.carv, self.data.steerAng]), traj_lowlevel_rel))
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
            state = np.array(state) #어레이 변환
            state, reward_argument, info = self.data.manage_state(state, action)

        # 리워드 계산
        reward = self.getReward(reward_argument, time)
        info = info

        if self.check == 0:
            self.data.render()

        return state, reward, done, info

    def getReward(self, reward_argument, time):
        traj = reward_argument['traj']
        caryaw = reward_argument['caryaw']
        traj_shape = Point(traj[0], traj[1])

        forbidden_area = [self.road.forbbiden_area1, self.road.forbbiden_area2, self.cone.cones_shape]
        forbidden_reward = - self.is_traj_in_forbidden_area(traj_shape) * 5000

        cone_reward = - self.is_traj_in_cone(traj_shape) * 5000

        axis_reward = - abs(traj[1] + 10) * 100
        yaw_reward = abs(caryaw) * 300
        e = forbidden_reward + cone_reward + axis_reward + yaw_reward
        if self.test_num % 100 == 0:
            print(f"[traj: {traj}] [forbidden: {forbidden_reward}] [cone: {cone_reward}] [Axis: {axis_reward}]")
        return e

    def is_traj_in_cone(self, traj_shape):
        for cone in self.cone.cones_shape:
            if traj_shape.intersects(cone):
                return 1
        return 0

    def is_traj_in_forbidden_area(self, traj_shape):
        if self.road.forbbiden_area1.intersects(traj_shape):
            return 1
        if self.road.forbbiden_area2.intersects(traj_shape):
            return 1
        return 0

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
