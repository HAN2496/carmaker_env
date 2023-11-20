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
from carmaker_env_low import CarMakerEnv as LowLevelCarMakerEnv
from stable_baselines3 import PPO, SAC
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, Point, LineString
from SLALOM_cone import Road, Car, Cone
import pygame
from carmaker_data import Data, Trajectory
from common_functions import *

# 카메이커 컨트롤 노드 구동을 위한 쓰레드
# CMcontrolNode 내의 sim_start에서 while loop로 통신을 처리하므로, 강화학습 프로세스와 분리를 위해 별도 쓰레드로 관리


class CarMakerEnvB(gym.Env):
    def __init__(self, road_type, check=2, use_low=False):
        # Action과 State의 크기 및 형태를 정의.
        matlab_path = 'C:/CM_Projects/JX1_102/src_cm4sl'
        host = '127.0.0.1'

        self.check = check
        self.use_low = False
        self.road_type = road_type
        self.data = Data(road_type=road_type, low_env=use_low, check=check)
        self.traj = Trajectory(road_type=road_type, low_env=use_low)
        self.last_carx = self.data.carx
        self.dist = 12

        env_action_num = np.size(self.data.manage_state_b())
        env_obs_num = 15

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(env_action_num,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env_obs_num,), dtype=np.float32)

    def _initial_state(self):
        self.test_num = 0
        return np.zeros(self.observation_space.shape)

    def reset(self):
        self.data._init()
        # 초기화 코드
        return self._initial_state()

    def step(self, action):

        state = np.zeros(np.size(self.data.manage_state_b()))
        self.test_num += 1
        done = False
        tmp = 0

        blevel_action = action[0]
        self.traj.update_b(self.data.carx, self.data.cary, self.data.caryaw, blevel_action)
        self.last_carx = self.data.carx
        traj_last_point = self.traj.get_ctrl_last_point()
        self.dist = np.linalg.norm(traj_last_point - np.array([self.data.carx, self.data.cary]))

        # 리워드 계산
        reward = self.data.manage_reward_b()
        info_key = np.array(["num", "time", "x", "y", "yaw", "carv", "ang", "vel", "acc", "devDist", "devAng",
                             "alHori", "roll", "rl", "rr", "fl", "fr"])
        info = {key: value for key, value in zip(info_key, self.data.simul_data)}

        return state, reward, done, info


if __name__ == "__main__":
    # 환경 테스트
    road_type = "SLALOM"
    env = CarMakerEnvB(road_type=road_type, check=0)
    act_lst = []
    next_state_lst = []
    info_lst = []


    for i in range(3):
        # 환경 초기화
        state = env.reset()

        # 에피소드 실행
        done = False
        try:
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

        except KeyboardInterrupt:
                print('Exit by Keyboard Interrupt')
