import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import re

CONER = 0.2
CARWIDTH, CARLENGTH = 1.8, 4

def load_data(type, comment=0):
    data = {}
    if comment == 0:
        data['info'] = pd.read_csv(f'{type}_info.csv')
        data['reward'] = pd.read_csv(f'{type}_reward.csv').loc[:, "0"].values
    else:
        data['info'] = pd.read_csv(f'{type}_{comment}_info.csv')
        data['reward'] = pd.read_csv(f'{type}_{comment}_reward.csv').loc[:, "0"].values

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

    roll_rate = np.sum(np.abs(np.diff(data_dict['roll']))) / time
    yaw_rate = np.sum(np.abs(np.diff(data_dict["caryaw"]))) / time
    maximum_lateral_acc = np.max(np.abs(data_dict['alHori']))
    total_reward = np.sum(data_dict['reward'])

    return [dataset, time,  roll_rate, yaw_rate, maximum_lateral_acc, total_reward]



def plot_trajectory(traj, ipg, rl):
#    plt.plot(traj[:, 0], traj[:, 1], label="Trjaectory", color='orange')
    plt.plot(ipg['carx'], ipg['cary'], label="IPG", color='blue')
    plt.plot(rl['carx'], rl['cary'], label="RL")
#    plt.axis("equal")
    plt.legend()
    plt.show()

def shape_car(carx, cary, caryaw):
    half_length = 2
    half_width = 0.9

    corners = [
        (-half_length, -half_width),
        (-half_length, half_width),
        (half_length, half_width),
        (half_length, -half_width)
    ]

    car_shape = Polygon(corners)
    car_shape = affinity.rotate(car_shape, caryaw, origin='center', use_radians=True)
    car_shape = affinity.translate(car_shape, carx, cary)

    return car_shape

def check_collision(cones, ipg):
    cones_shape = [Point(conex, coney).buffer(0.2) for conex, coney in cones]
    ipg_shape = [shape_car(carx, cary, caryaw) for carx, cary, caryaw in zip(ipg["carx"], ipg["cary"], ipg["caryaw"])]
    collisions = []

    for car in ipg_shape:
        for cone, (conex, coney) in zip(cones_shape, cones):
            if car.intersects(cone):
                collision_info = (conex, coney)
                if collision_info not in collisions:
                    collisions.append(collision_info)
    return collisions

