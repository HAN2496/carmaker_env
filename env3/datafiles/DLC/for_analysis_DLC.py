import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

CONER = 0.2
CARWIDTH, CARLENGTH = 1.8, 4
def load_data(prefix):
    data = {}
    data['info'] = pd.read_csv(f'{prefix}_info.csv')
    data['reward'] = pd.read_csv(f'{prefix}_reward.csv').loc[:, "0"].values

    extracted = {}
    df = data['info']
    extracted['time'] = df.loc[:, 'time'].values
    extracted['ang'] = df.loc[:, 'ang'].values
    extracted['vel'] = df.loc[:, 'vel'].values
    extracted['acc'] = df.loc[:, 'acc'].values
    extracted['alHori'] = df.loc[:, 'alHori'].values
    extracted['carv'] = df.loc[:, "carv"].values[:-1]
    extracted['carx'] = df.loc[:, 'x'].values[:-1]
    extracted['cary'] = df.loc[:, "y"].values[:-1]
    extracted['caryaw'] = df.loc[:, "yaw"].values[:-1]
    extracted['roll'] = df.loc[:, "roll"].values[:-1]

    extracted['reward'] = data['reward']
    return extracted


def plot_compare(x_data_list, title, idx, labels, subplot_shape):
    plt.subplot(*subplot_shape, idx)
    for i, x_data in enumerate(x_data_list):
        plt.plot(x_data, label=labels[i])
    plt.title(title)
    plt.legend()

def plot_multiple(data_keys, titles, labels, ipg, rl):
    num_plots = len(data_keys)
    subplot_shape = (np.ceil(num_plots / 3).astype(int), 3)

    for idx, (data_key, title) in enumerate(zip(data_keys, titles), start=1):
        plot_compare([ipg[data_key], rl[data_key]], title, idx, labels, subplot_shape)

    plt.tight_layout()
    plt.show()

def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1))

def get_value_or_interpolate(carx, carv, target_x):
    # Try to get the value at target_x
    if target_x in carx:
        return carv[np.where(carx == target_x)[0][0]]

    # If not, find indices of surrounding values
    greater_indices = np.where(carx > target_x)
    if not greater_indices[0].size:
        # If there are no greater values, return the last value
        return carv[-1]
    right_idx = greater_indices[0][0]
    left_idx = right_idx - 1

    return linear_interpolation(carx[left_idx], carv[left_idx], carx[right_idx], carv[right_idx], target_x)

def calc_performance(dataset, data_dict):

    time = data_dict['time'][-2]
    avg_carv = np.sum(np.abs(data_dict['carv'])) / time

    initial_carv = get_value_or_interpolate(data_dict['carx'], data_dict['carv'], 50)
    escape_carv = get_value_or_interpolate(data_dict['carx'], data_dict['carv'], 111)

    roll_rate = np.sum(np.abs(np.diff(data_dict['roll']))) / time
    yaw_rate = np.sum(np.abs(np.diff(data_dict["caryaw"]))) / time
    maximum_lateral_acc = np.max(np.abs(data_dict['alHori']))
    total_reward = np.sum(data_dict['reward'])

    return [dataset, time, initial_carv, escape_carv, roll_rate, yaw_rate, maximum_lateral_acc, total_reward]
