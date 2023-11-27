"""
Pretrain 데이터 저장하는 코드
"""

import gym
import numpy as np
import pandas as pd

data = np.load('expert_data.npz', allow_pickle=True)
observations = data['observations']
actions = data['actions']
rewards = data['rewards']
dones = data['dones']
infos = data['infos']

print(len(actions))
