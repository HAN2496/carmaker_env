"""
Pretrain 데이터 저장하는 코드
"""

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer
from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank,road_type, seed=0):

    def _init():
        #IPG로 돌리기
        env = CarMakerEnv(road_type=road_type, simul_path='test_IPG', port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"

    prefix = 'pretrain'

    env = make_env(0, road_type=road_type)()

    print("Program Start.\n")

    #expert 학습 시작
    expert_model = SAC('MlpPolicy', env, verbose=1)
    expert_model.learn(total_timesteps=100)

    print("Expert learning finished. Expert data will be collected.")

    # expert 데이터 수집
    expert_actions = []
    expert_observations = []
    expert_reward = []
    expert_done = []
    expert_info = []
    obs = env.reset()

    buffer_size = 100000
    for _ in range(buffer_size):
        action = expert_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action[0])
        expert_observations.append(obs)
        expert_reward.append(reward)
        expert_done.append(done)
        expert_actions.append(action[0])
        expert_info.append(info)
        if done:
            obs = env.reset()

    np.savez('expert_data.npz',
             buffer_size=buffer_size,
             observations=np.array(expert_observations),
             actions=np.array(expert_actions),
             rewards=np.array(expert_reward),
             dones=np.array(expert_done),
             infos=expert_info)

    print("Expert data saved.")

if __name__ == '__main__':
    main()