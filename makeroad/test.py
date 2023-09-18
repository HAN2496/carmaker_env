import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
from Make_env import MakeRoadEnv

env = MakeRoadEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.load(total_timesteps=10000)
