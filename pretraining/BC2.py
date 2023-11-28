from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout, types
import torch
import pandas as pd

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



for idx in range(data_len):
    data_idx = idx % data_len
    next_obs_data = observations[data_idx + 1] if data_idx < data_len - 1 else observations[data_len - 1]
    next_obs.append(next_obs_data)
next_observations = np.array(next_obs)


new_observations = []
new_actions = []
new_rewards = []
new_dones = []
new_infos = []
new_next_observation = []

data_len = len(observations)
for idx in range(buffer_size):
    data_idx = idx % data_len
    new_observations.append(observations[data_idx])
    new_actions.append(actions[data_idx])
    new_rewards.append(rewards[data_idx])
    new_dones.append(dones[data_idx])
    new_infos.append(infos[data_idx])
    new_next_observation.append(new_observations[data_idx])




transitions = types.Transitions(
    obs=observations,
    acts=actions,
    infos=infos,
    dones=dones,
    next_obs=next_observations
)

env = make_env(0, road_type=road_type)()
vec_env = DummyVecEnv([lambda: env])

input("Program Start.\n")

rng = np.random.default_rng(0)
model = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng
)

model.train(n_epochs=10 * 100,
            reset_tensorboard=True)


# 훈련된 모델을 사용하여 환경 테스트
print("Now test START!")
obs = vec_env.reset()
action_lst = []
reward_lst = []
info_lst = []
while True:
    action, _ = model.policy.predict(obs, deterministic=True)
    observation, reward, terminated, truncated, info = vec_env.step(action)
    info_lst.append(info)
    action_lst.append(action)
    reward_lst.append(reward)
    if terminated:
        df1 = pd.DataFrame(data=reward_lst)
        df1.to_csv(f'datafiles/pretrain_test/reward.csv')
        df3 = pd.DataFrame(data=info_lst)
        df3.to_csv(f'datafiles/pretrain_test/_info.csv', index=False)
        df4 = pd.DataFrame(data=action_lst)
        df4.to_csv(f'datafiles/pretrain_test/_action.csv', index=False)
        print("Episode Finished. Data saved.")
        obs = vec_env.reset()
        break

vec_env.close()