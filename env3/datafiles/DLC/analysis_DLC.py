import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from for_analysis_DLC import *

CONER = 0.2
CARWIDTH, CARLENGTH = 1.8, 4

#수정하는 부분
road_types = 'DLC'
df_traj = pd.read_csv(f'datasets_traj.csv')

ipg = load_data('IPG')
rl = load_data('RL')
traj_tx = df_traj.loc[:, "traj_tx"].values
traj_ty = df_traj.loc[:, "traj_ty"].values
labels = ['ipg', 'rl']

compare_keys = ['ang', 'vel', 'acc', 'carx', 'cary', 'reward']
titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "Car pos X", "Car pos Y", "Reward"]

plot_multiple(compare_keys, titles, labels, ipg, rl)
