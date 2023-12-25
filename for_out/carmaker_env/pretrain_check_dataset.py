from carmaker_env_low import CarMakerEnv
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
import os

data = np.load('expert_data.npz', allow_pickle=True)

observations = data['observations']
next_observations = data['next_observations']
actions = data['actions']
rewards = data['rewards']
dones = data['dones']
infos = data['infos']

print(np.shape(observations))
print(np.shape(next_observations))
print(np.shape(actions))