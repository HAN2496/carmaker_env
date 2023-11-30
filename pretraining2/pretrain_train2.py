import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from carmaker_env_low_pretrain import CarMakerEnv

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from torch.utils.data import Dataset, DataLoader
from callbacks import getBestRewardCallback

import torch
class Args:
    def __init__(self, prefix, alg):
        self.prefix = prefix
        self.alg = alg

def make_env(rank,road_type, seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, simul_path='pythonCtrl_JX1', port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"

    comment = 'pretrain'
    prefix = road_type + "/" + comment
    args = Args(prefix=prefix, alg='sac')

    bestRewardCallback = getBestRewardCallback(args)

    env = make_env(0, road_type=road_type)()
    env = Monitor(env, f"models/{prefix}")

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(f"models/{prefix}/tensorboard"))

    input("Program Start.\n")

    #저장된 데이터 불러오기
    data = np.load('expert_data.npz', allow_pickle=True)
    buffer_size = 10 * 10000
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    infos = data['infos']

    # 리플레이 버퍼 초기화 및 데이터 추가
    replay_buffer = ReplayBuffer(buffer_size=buffer_size, observation_space=env.observation_space,
                                 action_space=env.action_space)

    data_len = len(observations)
    for idx in range(buffer_size):
        data_idx = idx % data_len
        next_obs = observations[data_idx + 1] if data_idx < data_len - 1 else observations[data_len - 1]
        replay_buffer.add(observations[data_idx], next_obs, actions[data_idx], rewards[data_idx], dones[data_idx],
                          [infos[data_idx]])


    # 모델의 리플레이 버퍼를 사전 로드된 데이터로 설정
    model.replay_buffer = replay_buffer

    model.learn(total_timesteps=300 * 10000, callback=bestRewardCallback)
    model.save(f"models/{prefix}_last.pkl")

if __name__ == '__main__':
    main()
