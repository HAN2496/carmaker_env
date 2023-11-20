import time
from cm_control import CMcontrolNode
import gym
from gym import spaces
import threading
from queue import Queue
from stable_baselines3 import PPO, SAC
from carmaker_data import Data, Trajectory
from common_functions import *
from carmaker_env_low import CarMakerEnv as LowLevelCarMakerEnv
from carmaker_env_b import CarMakerEnvB as MetaController
from carmaker_env_a import CarMakerEnvA as SubController
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
class Agent(gym.Env):
    def __init__(self, road_type, check=2, port=10001, simul_path='pythonCtrl_JX1', use_low=False):
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

        #env에서는 1개의 action, simulink는 connect를 위해 1개가 추가됨
        env_action_num = np.size(self.data.manage_state_b())
        sim_action_num = env_action_num + 1

        # Env의 observation 개수와 simulink observation 개수
        env_obs_num = 15
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

        low_level_env = LowLevelCarMakerEnv(road_type=road_type, check=check, use_low=use_low)
        self.low_level_model = SAC.load(f"best_model/SLALOM_best_model.pkl", env=low_level_env)
        self.low_level_obs = low_level_env.reset()

        self.meta_controller = MetaController(road_type=road_type, check=check, use_low=False)
        self.sub_controller = SubController(road_type=road_type, check=check, use_low=False)

    def train(self, meta_timesteps, sub_timesteps):
        # 메타 컨트롤러 학습
        self.meta_controller.learn(meta_timesteps)

        # 하위 컨트롤러 학습
        self.sub_controller.learn(sub_timesteps)

    def run(self, total_steps):
        obs = self.env.reset()
        for _ in range(total_steps):
            # 메타 컨트롤러로부터 하위 목표 선택
                subgoal = self.meta_controller.select_subgoal(obs)

            # 하위 컨트롤러로부터 행동 선택
            action = self.sub_controller.take_action(obs)

            # 환경에 행동 적용
            obs, reward, done, info = self.env.step(action)

            # 에피소드 리셋
            if done:
                obs = self.env.reset()

if __name__ == "__main__":
    pass