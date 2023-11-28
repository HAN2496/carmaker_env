import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from carmaker_env_low import CarMakerEnv

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from callbacks import getBestRewardCallback

from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.algorithms.bc import BC
from imitation.data import rollout, types
import torch

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

road_type = "DLC"
comment = 'pretrain'

expert_data = np.load('expert_data.npz', allow_pickle=True)
buffer_size = 10 * 10000
observations = expert_data['observations']
actions = expert_data['actions']
rewards = expert_data['rewards']
dones = expert_data['dones']
infos = expert_data['infos']

data_len = len(observations)
next_obs = []



for idx in range(2019):
    data_idx = idx % data_len
    next_obs_data = observations[data_idx + 1] if data_idx < data_len - 1 else observations[data_len - 1]
    next_obs.append(next_obs_data)
next_observations = np.array(next_obs)

transitions = types.Transitions(
    obs=observations,
    acts=actions,
    infos=infos,
    dones=dones,
    next_obs=next_observations
)

env = make_env(0, road_type=road_type)()
vec_env = DummyVecEnv([lambda: env])

rng = np.random.default_rng(0)
model = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng
)

model.train(n_epochs=10)


# 훈련된 모델을 사용하여 환경 테스트
print("Now test START!")
obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.policy.predict(obs, deterministic=True)
    obs, _, done, _ = vec_env.step(action)
    if done:
        obs = vec_env.reset()