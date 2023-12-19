"""
강화학습에 사용할 수 있는 Gym Env 기반 카메이커 연동 클라스
cm_control.py에 구현된 기능을 이용
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from cm_control import CMcontrolNode
import threading
from queue import Queue
import pandas as pd
import time
from carmaker_cone import *
from carmaker_data import *

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

class CarMakerEnv(gymnasium.Env):
    def __init__(self, road_type, env_num=2, port=10001, simul_path='pythonCtrl_JX1', use_carmaker=True, show=False):
        matlab_path = 'C:/CM_Projects/JX1_102/src_cm4sl'
        host = '127.0.0.1'

        self.env_num = env_num
        self.use_carmaker = use_carmaker
        self.road_type = road_type
        self.show = show
        self.data = Data(road_type=road_type, env_num=env_num, show=show)

        env_action_num = 1
        sim_action_num = env_action_num + 1

        env_obs_num = np.size(self.data.manage_state_low())
        sim_obs_num = 17

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

    def __del__(self):
        self.cm_thread.join()

    def _initial_state(self):
        self.data._init()
        self.test_num = 0
        return np.zeros(self.observation_space.shape)

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None):
        if seed:
            self.seed(seed)

        if self.env_num == 0 and self.show:
            self.data.render()

        if self.use_carmaker:
            if self.sim_initiated == True:
                # 한번의 시뮬레이션도 실행하지 않은 상태에서는 stop 명령을 줄 필요가 없음
                self.status_queue.put("stop")
                self.action_queue.queue.clear()
                self.state_queue.queue.clear()

            self.sim_started = False

        return self._initial_state(), {}

    def step(self, action1):
        action = np.append(self.test_num, action1)
        self.test_num += 1

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
            self.data.put_simul_data(state)
            state = self.data.manage_state_low()

        if state.any() == False:
            reward = 0.0

        # 리워드 계산
        reward = self.data.manage_reward_low()
        info_key = np.array(["num", "time", "x", "y", "yaw", "carv", "ang", "vel", "acc", "devDist", "devAng",
                             "alHori", "roll", "rl", "rr", "fl", "fr"])
        info = {key: value for key, value in zip(info_key, self.data.simul_data)}
        if self.show and self.env_num == 0:
            self.data.render()

        truncated = False
        return state, reward, done, truncated, info


    def getReward(self, time):

        if state.any() == False:
            # 에피소드 종료시
            return 0.0

        return self.data.manage_reward_low()


if __name__ == "__main__":
    # 환경 테스트
    env = CarMakerEnv(road_type="DLC", simul_path="test_IPG", env_num=0, show=True)
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
            next_state, reward, done, _, info = env.step(action)

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
