import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import re

CONER = 0.2
CARWIDTH, CARLENGTH = 1.568, 4.3
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

def plot_multiple(data_keys, titles, labels, *datasets):
    num_plots = len(data_keys)
    subplot_shape = (int(np.ceil(num_plots / 3)), 3)  # np.ceil의 결과는 float이므로 int로 변환

    for idx, (data_key, title) in enumerate(zip(data_keys, titles), start=1):
        # 각 데이터셋에서 data_key에 해당하는 데이터를 추출
        data_list = [dataset.get(data_key) for dataset in datasets]
        plot_compare(data_list, title, idx, labels, subplot_shape)

    plt.tight_layout()
    plt.show()

def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1))

def get_value_or_interpolate(carx, carv, target_x):
    if target_x in carx:
        return carv[np.where(carx == target_x)[0][0]]

    greater_indices = np.where(carx > target_x)
    if not greater_indices[0].size:
        return carv[-1]
    right_idx = greater_indices[0][0]
    left_idx = right_idx - 1

    return linear_interpolation(carx[left_idx], carv[left_idx], carx[right_idx], carv[right_idx], target_x)

def calc_performance(dataset, data_dict):
    rad2deg = 57.2958
    ms2kph = 3.6

    time = data_dict['time'][-2]
    avg_carv = np.sum(np.abs(data_dict['carv'])) / time

    initial_carv = get_value_or_interpolate(data_dict['carx'], data_dict['carv'], 52) * ms2kph
    escape_carv = get_value_or_interpolate(data_dict['carx'], data_dict['carv'], 111) * ms2kph

    roll_rate = np.sum(np.abs(np.diff(data_dict['roll']))) / time
    yaw_rate = np.sum(np.abs(np.diff(data_dict["caryaw"]))) / time * rad2deg
    maximum_lateral_acc = np.max(np.abs(data_dict['alHori']))

    total_reward = np.sum(data_dict['reward'])

    return [dataset, time, initial_carv, escape_carv, roll_rate, yaw_rate, maximum_lateral_acc, total_reward]





def plot_trajectory(cones, traj, ipg, rl, labels):
    plt.scatter(cones[:, 0], cones[:, 1], label='Cone', color='red', linewidth=3)
#    plt.plot(traj[:, 0], traj[:, 1], label="Trjaectory", color='orange')
    plt.plot(ipg['carx'], ipg['cary'], label="IPG", color='blue', linewidth=3)
    plt.plot(rl['carx'], rl['cary'], label=labels, linewidth=3)
#    plt.axis("equal")
    plt.xlabel("m")
    plt.ylabel("m")
    plt.title("Trajectory")
    plt.legend()
    plt.show()

def shape_car(carx, cary, caryaw):
    half_length = CARLENGTH / 2
    half_width = CARWIDTH / 2

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

