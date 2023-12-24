import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from for_analysis import *

CONER = 0.2
CARWIDTH, CARLENGTH = 1.975, 4.94

#수정하는 부분
road_types = 'Total'
traj = pd.read_csv(f'datasets_traj.csv').loc[:, ["traj_tx", "traj_ty"]].values

ipg = load_data('IPG', comment=0)
rl = load_data('total_mpc', comment=0)
labels = ['ipg', 'mpc']

compare_keys = ['ang', 'vel', 'acc', 'carx', 'cary', 'reward']
titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "Car pos X", "Car pos Y", "Reward"]

#plot_multiple(compare_keys, titles, labels, ipg, rl)
plot_trajectory(traj, ipg, rl)

tables = []
for dataset, data_dict in zip(labels, [ipg, rl]):
    calc_data = calc_performance(dataset, data_dict)
    tables.append(calc_data)

comparsion_row = ["Comparision (%)"]
for col1, col2 in zip(tables[0][1:], tables[1][1:]):
    comp = (col2 - col1) / col1 * 100
    comparsion_row.append(comp)
tables.append(comparsion_row)

col = ['name', 'Time',  'roll rate', 'yaw rate', 'maxium lateral acc', 'total reward']
df = pd.DataFrame(tables, columns=col)
print(df)


#회전반경

def create_DLC_cone_arr():
    sections = [
        {'start': 150, 'gap': 3, 'cone_dist': 2.4225, 'num': 5, 'y_offset': -12.25},
        {'start': 175, 'gap': 11.5/4, 'cone_dist': 2.975, 'num': 5, 'y_offset': -12.25+2.61125},  #
        {'start': 199, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -12.25-0.28875}
    ]
    cones = []

    for section in sections:
        for i in range(section['num']):  # Each section has 5 pairs
            x_base = section['start'] + section['gap'] * i
            y1 = section['y_offset'] - section['cone_dist'] / 2
            y2 = section['y_offset'] + section['cone_dist'] / 2
            cones.extend([[x_base, y1], [x_base, y2]])

    return np.array(cones)

print(check_collision(create_DLC_cone_arr(), ipg))
print(check_collision(create_DLC_cone_arr(), rl))
