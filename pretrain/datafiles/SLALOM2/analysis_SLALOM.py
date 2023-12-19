import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from for_analysis_SLALOM import *

CONER = 0.2
CARWIDTH, CARLENGTH = 1.8, 4

cones = np.array([[100 + 30 * i, -25] for i in range(10)])

#수정하는 부분
road_types = 'SLALOM2'
traj = pd.read_csv(f'datasets_traj.csv').loc[:, ["traj_tx", "traj_ty"]].values

ipg = load_data('IPG', comment='rws')
rl = load_data('RL', comment='rws')
labels = ['ipg', 'rl']

compare_keys = ['ang', 'vel', 'acc', 'carx', 'cary', 'reward']
titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "Car pos X", "Car pos Y", "Reward"]

plot_multiple(compare_keys, titles, labels, ipg, rl)
plot_trajectory(cones, traj, ipg, rl)
print(check_collision(cones, ipg))

tables = []
for dataset, data_dict in zip(labels, [ipg, rl]):
    calc_data = calc_performance(dataset, data_dict)
    tables.append(calc_data)

comparsion_row = ["Comparision (%)"]
for col1, col2 in zip(tables[0][1:], tables[1][1:]):
    comp = (col2 - col1) / col1 * 100
    comparsion_row.append(comp)
tables.append(comparsion_row)

col = ['name', 'Time', 'inital carv', 'escape carv', 'roll rate', 'yaw rate', 'maxium lateral acc', 'total reward']
df = pd.DataFrame(tables, columns=col)
print(df)
