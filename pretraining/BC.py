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
rng = np.random.default_rng(0)
env = make_env(0, road_type=road_type)()
vec_env = DummyVecEnv([lambda: env])

#expert_model = SAC("MlpPolicy", vec_env, verbose=1)

expert_model = SAC.load(f"best_model/DLC_best_model.pkl", env=vec_env)
transitions = rollout.generate_transitions(expert_model, vec_env, n_timesteps=2, rng=rng)
dataset = rollout.flatten_trajectories(transitions)

# Step 3: Behavioral Cloning을 통한 학습
print("Now BC train START!")
bc_model = BC("MlpPolicy", dataset, vec_env)
bc_model.train(n_epochs=10)

# 훈련된 모델을 사용하여 환경 테스트
print("Now test START!")
obs = vec_env.reset()
for _ in range(1000):
    action, _ = bc_model.policy.predict(obs, deterministic=True)
    obs, _, done, _ = vec_env.step(action)
    if done:
        obs = vec_env.reset()