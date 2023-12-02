"""
학습 코드 예제
1. 카메이커 연동 환경을 불러온다
    1-1. 여러 대의 카메이커를 실행하기 위해 SubprocVecEnv를 이용하여 멀티프로세싱이 가능한 환경 로드
2. 학습에 사용할 RL 모델(e.g. PPO)을 불러온다.
3. 학습을 진행한다. x
    3-1. total_timesteps 수를 변화시켜 충분히 학습하도록 한다.
4. 학습이 완료된 후 웨이트 파일(e.g. model.pkl)을 저장한다.
"""
import sys
from carmaker_env_low import CarMakerEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from callbacks import getBestRewardCallback, logDir, rmsLogging
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
import os
import torch
import logging
from datetime import datetime

logging.basicConfig(filename='Log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
custom_logger = logging.getLogger('customLogger')
custom_logger.propagate = False
handler = logging.FileHandler('Log.txt')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
custom_logger.addHandler(handler)
custom_logger.setLevel(logging.INFO)

# GPU를 사용할 수 있는지 확인합니다.
if torch.cuda.is_available():
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device("cuda:" + str(device_id))
    print(f"Using GPU device ID {device_id}.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

class Args:
    def __init__(self, prefix, alg):
        self.prefix = prefix
        self.alg = alg

def make_env(rank, road_type, seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, port=10000 + rank, env_num=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

def main():
    """
    변경할 부분
    1. env_num: env_{road_tpye}_{num}의 num과 동일하게 맞춰줄 것
    2. road_type: 현재 도로 입력할 것. DLC, SLALOM, UTurn
    3. 위의 import에 제대로 된 env를 불러왔는지 확인할 것
    4. 추가 설명 내용이 있을 경우 explanation에 글을 작성하면 Log.txt에 기록됨
    """

    road_type = "DLC"
    comment = "rws_dlc changed"
    explanation = "carmaker env low"

    num_proc = 5
    naming = f"env_{comment}"
    prefix = road_type + "/" + naming
    args = Args(prefix=prefix, alg='sac')

    bestRewardCallback = getBestRewardCallback(args)

    env = SubprocVecEnv([make_env(i, road_type=road_type) for i in range(num_proc)])
    env = VecMonitor(env, f"models/{prefix}")

    input("Program Start.\n")

    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=os.path.join(f"models/{prefix}/tensorboard"))
#    model = PPO.load("UTurn_env22_best_model.pkl", env=env, verbos=1, tensorboard_log=os.path.join("tensorboard/{}/Additional_study".format(naming)))

    try:
        custom_logger.info(f"[{prefix}]")
        custom_logger.info(f" --> {explanation}")
        logging.info(f"{prefix} - Training Start")
        model.learn(total_timesteps=10000*300, log_interval=50, callback=bestRewardCallback)


    except KeyboardInterrupt:
        logging.info(f"{prefix} - Keyboard Interrupt")
        print("Learning interrupted. Not save model.")

    finally:
        print("Saving model..")
        logging.info(f"{prefix} - Training End")
        model.save(f"models/{prefix}_last.pkl")
        print("Model saved.")


if __name__ == '__main__':
    main()
