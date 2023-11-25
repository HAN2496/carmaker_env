"""
Pretrain 데이터 저장하는 코드
"""

import gym
import numpy as np
import pandas as pd

data = np.load('expert_data_preprocessing.npz', allow_pickle=True)
actions = data['actions'][1900:]
infos = data['infos'][1900:]
print(np.size(actions), np.size(infos))
print("***")
for i in range(10):
    print(actions[i])

print("__________")
before_ang = 0
tmp=[]
for idx, info in enumerate(infos):
    tmp.append(idx)
    if 0 <= idx <= 10:

        ang = info['ang']
        print(info)
        print(info['num'], info['ang'])
        print("&&")
        print((ang - before_ang) / 0.15)
        before_ang = ang
