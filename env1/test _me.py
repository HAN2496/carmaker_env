import numpy as np


a = np.array([0, 0])
b = np.array([3, 4])

traj_reward = np.linalg.norm(b-a)

print(traj_reward)