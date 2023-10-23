import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from for_analysis_SLALOM import *

CONER = 0.2
CARWIDTH, CARLENGTH = 1.568, 4.3

cones = np.array([[100 + 30 * i, -10] for i in range(10)])

#수정하는 부분
road_types = 'SLALOM'
traj = pd.read_csv(f'datasets_traj_SLALOM.csv').loc[:, ["traj_tx", "traj_ty"]].values

ipg = load_data('IPG', 'cutting1')
rl = load_data('RL', "env1")
mpc = load_data('MPC')
labels = ['ipg', 'rl', 'mpc']

compare_keys = ['ang', 'vel', 'acc', 'carx', 'cary', 'reward']
titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "Car pos X", "Car pos Y", "Reward"]

plot_multiple(compare_keys, titles, labels, ipg, rl, mpc)
plot_trajectory(cones, traj, ipg, rl, mpc)
print("IPG: ", check_collision(cones, ipg))
print("RL: ",check_collision(cones, rl))
print("MPC: ",check_collision(cones, mpc))

tables = []
for dataset, data_dict in zip(labels, [ipg, rl, mpc]):
    calc_data = calc_performance(dataset, data_dict)
    tables.append(calc_data)

comparsion_row = ["Comparision ipg vs rl (%)"]
for col1, col2 in zip(tables[0][1:], tables[1][1:]):
    comp = (col2 - col1) / col1 * 100
    comparsion_row.append(comp)
tables.append(comparsion_row)

comparsion_row = ["Comparision ipg vs mpc (%)"]
for col1, col2 in zip(tables[0][1:], tables[2][1:]):
    comp = (col2 - col1) / col1 * 100
    comparsion_row.append(comp)
tables.append(comparsion_row)

col = ['name', 'Time', 'inital carv', 'escape carv', 'roll rate', 'yaw rate', 'maxium lateral acc', 'total reward']
df = pd.DataFrame(tables, columns=col)
print(df)
