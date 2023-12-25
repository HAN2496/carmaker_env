import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from for_analysis import *

CONER = 0.2
CARWIDTH, CARLENGTH = 1.8, 4

#수정하는 부분
road_types = 'Ramp'
traj = pd.read_csv(f'datasets_traj.csv').loc[:, ["traj_tx", "traj_ty"]].values

cut_start=600
cut_end = 10
for i in range(10):
    print("*"*50)
    cut_start = 500 + i * 10
    cut_end = 100 - i * 20
    print(f"cut start: {cut_start} / cut end: {cut_end}")
    print('\n')
    ipg = load_data('IPG', comment=0, cut_start=cut_start, cut_end=cut_end)
    rl = load_data('ramp_mpc', comment=0, cut_start=cut_start, cut_end=cut_end)
    labels = ['ipg', 'mpc-rl']

    compare_keys = ['ang', 'vel', 'acc', 'caryaw', 'alHori', 'roll']
    titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "caryaw", "alHori", "Roll"]

    #plot_multiple(compare_keys, titles, labels, ipg, rl)
    #plot_trajectory(traj, ipg, rl)

    tables = []
    for dataset, data_dict in zip(labels, [ipg, rl]):
        calc_data = calc_performance(dataset, data_dict)
        tables.append(calc_data)

    comparsion_row = ["Comparision (%)"]
    for col1, col2 in zip(tables[0][1:], tables[1][1:]):
        comp = (col2 - col1) / col1 * 100
        comparsion_row.append(comp)
    tables.append(comparsion_row)

    col = ['name', 'Time',  'roll rate', 'yaw rate', 'maxium lateral acc','total distance', 'total reward']
    df = pd.DataFrame(tables, columns=col)
    print(df)
    print('\n')
    check_traj_dist(traj, ipg, rl)
    print('\n')
    #회전반경
    #print(calc_turning_radius(ipg, rl))
    print("*"*50)
