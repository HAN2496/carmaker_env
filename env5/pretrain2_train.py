import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer
from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank,road_type,  simul_path='pythonCtrl_JX1', seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, simul_path=simul_path, port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"

    prefix = 'pretrain'

    env = make_env(0, road_type=road_type, simul_path='test_IPG')()

    print("Program Start.\n")

    data = np.load('expert_data.npz', allow_pickle=True)
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    infos = data['infos']

    # 리플레이 버퍼 초기화 및 데이터 추가
    replay_buffer = ReplayBuffer(buffer_size=100000, observation_space=env.observation_space,
                                 action_space=env.action_space)

    for idx in range(len(observations) - 1):
        next_obs = observations[idx+1] if idx < len(observations) - 1 else None
        replay_buffer.add(observations[idx], observations[idx + 1], actions[idx], rewards[idx], dones[idx],
                          [infos[idx]])


    model = SAC("MlpPolicy", env, verbose=1)


    # 사전 학습된 데이터로 모델 초기화
    model.replay_buffer = replay_buffer

    # 모델 훈련
    model.learn(total_timesteps=100000)


if __name__ == '__main__':
    main()