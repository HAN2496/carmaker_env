import numpy as np
import os
import torch
import logging

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from carmaker_env_low import CarMakerEnv
from callbacks import getBestRewardCallback


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

def make_env(rank,road_type, seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, port=10000 + rank, env_num=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"
    comment = "rws_pretrain_100K"
    buffer_size = 10 * 10000
    pretrain_steps = 100 * 1000
    num_proc = 2

    prefix = road_type + "/" + comment
    args = Args(prefix=prefix, alg='sac')

    bestRewardCallback = getBestRewardCallback(args)

    env = SubprocVecEnv([make_env(i, road_type=road_type) for i in range(num_proc)])
    env = VecMonitor(env, f"models/{prefix}")

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(f"models/{prefix}/tensorboard"))

    input("Program Start.\n")

    #저장된 데이터 불러오기
    data = np.load('expert_data.npz', allow_pickle=True)
    observations = data['observations']
    next_observations = data['next_observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    infos = data['infos']

    # 리플레이 버퍼 초기화 및 데이터 추가
    replay_buffer = ReplayBuffer(buffer_size=buffer_size, observation_space=env.observation_space,
                                 action_space=env.action_space, n_envs=num_proc)

    #버퍼사이즈 만큼 반복해서 데이터 넣어주기.
    data_len = len(observations)
    for idx in range(buffer_size):
        data_idx = idx % data_len
        obs = np.array([observations[data_idx]] * num_proc)
        next_obs = np.array([next_observations[data_idx]] * num_proc)
        act = np.array([actions[data_idx]] * num_proc)
        rew = np.array([rewards[data_idx]] * num_proc)
        done = np.array([dones[data_idx]] * num_proc)
        info = [infos[data_idx]] * num_proc

        replay_buffer.add(obs, next_obs, act, rew, done, info)


    # 모델의 리플레이 버퍼에 넣어주기
    model.replay_buffer = replay_buffer

    #학습시작
    model.learn(total_timesteps=300 * 10000, callback=bestRewardCallback, pretrain_steps=pretrain_steps)
    model.save(f"models/{prefix}_last.pkl")

if __name__ == '__main__':
    main()
