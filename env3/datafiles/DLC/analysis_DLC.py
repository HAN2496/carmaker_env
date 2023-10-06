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
        arr.append([start[0] + i * interval, start[1] + width / 2])
        arr.append([start[0] + i * interval, start[1] - width / 2])
    return arr

cone1 = make_dlc_cone([50, -10], 3, 2.23, 5)
cone2 = make_dlc_cone([75.5, -6.485], 2.75, 2.8, 5)
cone3 = make_dlc_cone([99, -10.385], 3, 3, 5)
cones = np.array(cone1 + cone2 + cone3)


#수정하는 부분
road_types = 'DLC'
traj = pd.read_csv(f'datasets_traj.csv').loc[:, ["traj_tx", "traj_ty"]].values

ipg = load_data('IPG')
rl = load_data('RL', "test")
labels = ['ipg', 'rl']

compare_keys = ['ang', 'vel', 'acc', 'carx', 'cary', 'reward']
titles = ['Steering Angle', "Steering Velocity", "Steering Acceleration", "Car pos X", "Car pos Y", "Reward"]

#plot_multiple(compare_keys, titles, labels, ipg, rl)
plot_trajectory(cones, traj, ipg, rl)
print(check_collision(cones, ipg))
