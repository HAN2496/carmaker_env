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
import time
from carmaker_env_low import CarMakerEnv as LowLevelCarMakerEnv
from stable_baselines3 import PPO, SAC
from carmaker_data import Data, Trajectory
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
    def __init__(self, road_type, env_num=2, port=10001, simul_path='pythonCtrl_JX1'):
        # Action과 State의 크기 및 형태를 정의.
        matlab_path = 'C:/CM_Projects/JX1_102/src_cm4sl'
        host = '127.0.0.1'

        self.env_num = env_num
        self.road_type = road_type
        self.data = Data(road_type=road_type, env_num=env_num, show=True)
        self.traj_end = self.data.traj.end_point
        self.check_while = 0

        #env에서는 1개의 action, simulink는 connect를 위해 1개가 추가됨
        env_action_num = 2
        sim_action_num = 2

        # Env의 observation 개수와 simulink observation 개수
        env_obs_num = np.size(self.data.manage_state_b())
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

        low_level_env = LowLevelCarMakerEnv(road_type=road_type, use_carmaker=False, env_num=env_num)
        self.low_level_model = SAC.load(f"best_model/DLC_best_model.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()

    def __del__(self):
        self.cm_thread.join()

    def _initial_state(self):
        self.data._init()
        self.check_while = 0
        if self.env_num == 0:
            self.data.render()
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
        done = False
        b_done = False
        tmp = 0

        # 최초 실행시
        if self.sim_initiated == False:
            self.sim_initiated = True

        # 에피소드의 첫 스텝
        if self.sim_started == False:
            self.status_queue.put("start")
            self.sim_started = True


        for _ in range(10):

            #print("In while")
            #print(f"In while: {round(self.traj_end[0], 2)}, {round(self.data.carx, 2)}")
            self.low_level_obs = self.data.manage_state_low()
            steering_changes = self.low_level_model.predict(self.low_level_obs)
            action_to_sim = np.append(self.data.test_num, steering_changes[0])

            # Action 값 전송 / State 값 수신
            self.action_queue.put(action_to_sim)
            low_state = self.state_queue.get()

            if self.env_num == 0:
                self.data.render()

            if low_state == False:
                #print("DONE: in while")
                state = self._initial_state()
                done = True
                break
            else:
                low_state = np.array(low_state)  # 어레이 변환
                self.data.put_simul_data(low_state)
                if b_done:
                    #print("DONE: in Else")
                    state = self._initial_state()
                    done = True
                    break

        #print("Out while")

        self.data.traj.update_traj([self.data.carx, self.data.cary, self.data.caryaw], action)
        self.traj_end = self.data.traj.end_point

        state, reward, b_done, info = self.data.manage_b()

        if done:
           # print("DONE: Out while")
            done = True
            state = self._initial_state()
            reward = 0.0

        if self.env_num == 0:
            self.data.render()
        return state, reward, done, info


if __name__ == "__main__":
    # 환경 테스트
    road_type = "DLC"
    env = CarMakerEnvB(road_type=road_type, simul_path='pythonCtrl_JX1', env_num=0)
    act_lst = []
    next_state_lst = []
    info_lst = []


    for i in range(10):
        print(i)
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
