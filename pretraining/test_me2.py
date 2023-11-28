import numpy as np

expert_data = np.load('expert_data.npz', allow_pickle=True)
buffer_size = 10 * 10000
observations = expert_data['observations']
actions = expert_data['actions']
rewards = expert_data['rewards']
dones = expert_data['dones']
infos = expert_data['infos']

data_len = len(observations)
next_obs=[]
for idx in range(2019):
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

print(len(new_observations))
if new_observations[0] == new_observations[2019]:
    print('here')