import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from for_analysis_DLC import *

CONER = 0.2
CARWIDTH, CARLENGTH = 1.8, 4

def make_dlc_cone(start, interval, width, num):
    arr = []
    for i in range(num):
        arr.append([start[0] + i * interval, start[1] + width / 2 + CONER])
        arr.append([start[0] + i * interval, start[1] - width / 2 - CONER])
    return arr

cone1 = make_dlc_cone([50, -10], 3, 2.23, 5)
cone2 = make_dlc_cone([75.5, -6.485], 2.75, 2.8, 5)
cone3 = make_dlc_cone([99, -10.385], 3, 3, 5)
cones = np.array(cone1 + cone2 + cone3)


#수정하는 부분
road_types = 'DLC'
traj = pd.read_csv(f'datasets_traj.csv').loc[:, ["traj_tx", "traj_ty"]].values

ipg_rl = load_data('IPG', 'rws')
rl = load_data('RL', "rws1")
labels = ['ipg', 'rl']

compare_keys = ['ang', 'vel', 'acc', 'carx', 'cary', 'reward']
titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "Car pos X", "Car pos Y", "Reward"]

plot_multiple(compare_keys, titles, ['ipg', 'rl'], ipg_rl, rl)
plot_multiple(compare_keys, titles, ['ipg', 'mpc'], )
plot_trajectory(cones, traj, ipg_rl, rl, 'RL')

print("IPG: ", check_collision(cones))
print("RL: ",check_collision(cones, rl))

tables_rl = []
for dataset, data_dict in zip(['ipg', 'rl'], [ipg_rl, rl]):
    calc_data = calc_performance(dataset, data_dict)
    tables_rl.append(calc_data)

comparsion_row = ["Comparision ipg vs rl (%)"]
for col1, col2 in zip(tables_rl[0][1:], tables_rl[1][1:]):
    comp = (col2 - col1) / col1 * 100
    comparsion_row.append(comp)
tables_rl.append(comparsion_row)


col = ['name', 'Time', 'inital carv', 'escape carv', 'roll rate', 'yaw rate', 'maxium lateral acc', 'total reward']
df = pd.DataFrame(tables_rl, columns=col)
print(df)
