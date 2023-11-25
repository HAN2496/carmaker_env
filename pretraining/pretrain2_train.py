<<<<<<< HEAD
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer
from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.utils import set_random_seed
=======
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from callbacks import getBestRewardCallback

class Args:
    def __init__(self, prefix, alg):
        self.prefix = prefix
        self.alg = alg
>>>>>>> 9cb65c1806165dcf03ae7a413eca3b898f7fc0b1

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

    print("Program Start.\n")

    #저장된 데이터 불러오기
    data = np.load('expert_data_preprocessing.npz', allow_pickle=True)
    buffer_size = data['buffer_size']
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    infos = data['infos']

    # 리플레이 버퍼 초기화 및 데이터 추가
    replay_buffer = ReplayBuffer(buffer_size=buffer_size, observation_space=env.observation_space,
                                 action_space=env.action_space)

    for idx in range(len(observations) - 1):
        next_obs = observations[idx+1] if idx < len(observations) - 1 else None
        replay_buffer.add(observations[idx], next_obs, actions[idx], rewards[idx], dones[idx],
                          [infos[idx]])

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join(f"models/{prefix}/tensorboard"))

    # 사전 학습된 데이터로 모델 초기화
    model.replay_buffer = replay_buffer
    print('put All replay buffer')

    # 이제 모델 훈련
    model.learn(total_timesteps=300 * 10000, callback=bestRewardCallback)


if __name__ == '__main__':
    main()
