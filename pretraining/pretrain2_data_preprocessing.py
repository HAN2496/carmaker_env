"""
Pretrain 데이터 저장하는 코드
"""

import gym
import numpy as np


data = np.load('expert_data.npz', allow_pickle=True)
buffer_size = data['buffer_size'] - 1900
print(f"buffer size: {buffer_size}")
observations = data['observations'][1900:]
actions = data['actions'][1900:]
rewards = data['rewards'][1900:]
dones = data['dones'][1900:]
infos = data['infos'][1900:]
print(infos)
actions2 = []

idx2 = 0
ang1 = 0
for idx, info in enumerate(infos):

    if info['num'] == 0:
        idx2 =0

    if idx2 == 0:
        actions2.append(np.array([0]))
    elif idx2 == np.size(actions)-1:
        actions2.append(np.array([0]))
    else:
        ang1 = infos[idx2 - 1]['ang']
        ang2 = info['ang']

        action = (ang2 - ang1) / 0.15
        actions2.append(np.array([action]))

    idx2 += 1

actions2 = np.array(actions2)
print(type(actions), type(actions2))
print("-----------")
print(np.size(actions), np.size(actions2))
print("-----------")
print(actions)
print("-----------")
print(actions2)
np.savez('expert_data_preprocessing.npz',
         buffer_size=buffer_size,
         observations=np.array(observations),
         actions=np.array(actions2),
         rewards=np.array(rewards),
         dones=np.array(dones),
         infos=infos)
print("Saved")