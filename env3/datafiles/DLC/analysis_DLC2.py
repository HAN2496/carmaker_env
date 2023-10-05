import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from prettytable import PrettyTable

def str_to_float(df):
    new_df = []
    for i in df:
        if isinstance(i, str):
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", i)
            float_numbers = [np.float64(number) for number in numbers if number]
            new_df.extend(float_numbers)
    return np.array(new_df, dtype=np.float64)

def load_data(prefix, road_type):
    data = {}
    data['info'] = pd.read_csv(f'{prefix}_info.csv')
    data['reward'] = pd.read_csv(f'{prefix}_reward.csv').loc[:, "0"].values
    if prefix != 'datasets_traj':
        data['action'] = pd.read_csv(f'{prefix}_action.csv')
    return data

def extract_data(data, prefix):
    extracted = {}
    df = data['info']
    extracted['time'] = df.loc[:, 'Time'].values
    extracted['ang'] = df.loc[:, 'Steer.Ang'].values
    extracted['vel'] = df.loc[:, 'Steer.Vel'].values
    extracted['acc'] = df.loc[:, 'Steer.Acc'].values
    extracted['alHori'] = df.loc[:, 'alHori'].values
    extracted['carv'] = df.loc[:, "carv"].values[:-1]
    extracted['carx'] = df.loc[:, 'carx'].values[:-1]
    extracted['cary'] = df.loc[:, "cary"].values[:-1]
    extracted['caryaw'] = df.loc[:, "caryaw"].values[:-1]
    extracted['roll'] = df.loc[:, "Roll"].values[:-1]

    extracted['reward'] = data['reward']
    if prefix != 'datasets_traj':
        extracted['action'] = data['action']
    return extracted

#수정하는 부분
road_types = 'DLC'
df_traj = pd.read_csv(f'datasets_traj.csv')

ipg = load_data('IPG', road_types)
rl = load_data('RLWS_test', road_types)
ipg_data = extract_data(ipg, 'ipg')
rl_data = extract_data(rl, 'rl')
traj_tx = df_traj.loc[:, "traj_tx"].values
traj_ty = df_traj.loc[:, "traj_ty"].values
datasets = ['ipg', 'rl']
data_dicts = [ipg_data, rl_data]

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

#    roll_rate = np.sum(np.abs(data_dict['roll'])) / time
    roll_rate = np.sum(np.abs(np.diff(data_dict['roll']))) / time
#    yaw_rate = np.sum(np.abs(data_dict['caryaw'])) / time
    yaw_rate = np.sum(np.abs(np.diff(data_dict["caryaw"]))) / time
#    avg_lateral_acc = np.sum(np.abs(data_dict['alHori'])) / time
    maximum_lateral_acc = np.max(np.abs(data_dict['alHori']))
    total_reward = np.sum(data_dict['reward'])

    return [dataset, time, initial_carv, escape_carv, roll_rate, yaw_rate, maximum_lateral_acc, total_reward]

tables = []

for dataset, data_dict in zip(datasets, data_dicts):
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


def plot_compare(x_data_list, title, idx, labels):
    plt.subplot(2, 3, idx)
    for i, x_data in enumerate(x_data_list):
        plt.plot(x_data, label=labels[i])
    plt.title(title)
    plt.legend()


plot_compare([ipg_data['ang'], rl_data['ang']], "Steer Ang", 1, ['ipg', 'rl'])
plot_compare([ipg_data['vel'], rl_data['vel']], "Steer Acc", 3, ['ipg', 'rl'])
plot_compare([ipg_data['acc'], rl_data['acc']], "Steer Vel", 2, ['ipg', 'rl'])
plot_compare([ipg_data['carx'], rl_data['carx']], "Car Pos X", 4, ['ipg', 'rl'])
plot_compare([ipg_data['cary'], rl_data['cary']], "Car Pos Y", 5, ['ipg', 'rl'])
plot_compare([ipg_data['reward'], rl_data['reward']], "Reward", 6, ['ipg', 'rl'])

plt.show()

plot_compare([ipg_data['ang'], rl_data['ang']], "Yaw", 1, ['ipg', 'rl'])
plot_compare([ipg_data['ang'], rl_data['ang']], "alHori", 2, ['ipg', 'rl'])
plot_compare([ipg_data['ang'], rl_data['ang']], "Velocity", 3, ['ipg', 'rl'])

plt.show()


cones = [[50 + 3 * 0, -8.0525 + 0.9874, 11], [50, -8.0525 - 0.9874, 12],
         [50 + 3 * 1, -8.0525 + 0.9874, 21], [50 + 3 * 1, -8.0525 - 0.9874, 22],
         [50 + 3 * 2, -8.0525 + 0.9874, 31], [50 + 3 * 2, -8.0525 - 0.9874, 32],
         [50 + 3 * 3, -8.0525 + 0.9874, 41], [50 + 3 * 3, -8.0525 - 0.9874, 42],
         [50 + 3 * 4, -8.0525 + 0.9874, 51], [50 + 3 * 4, -8.0525 - 0.9874, 52],
         [75.5 + 3 * 0, -4.8315 + 1.26, 61], [75.5 + 3 * 0, -4.8315 - 1.26, 62],
         [75.5 + 3 * 1, -4.8315 + 1.26, 71], [75.5 + 3 * 1, -4.8315 - 1.26, 72],
         [75.5 + 3 * 2, -4.8315 + 1.26, 81], [75.5 + 3 * 2, -4.8315 - 1.26, 82],
         [75.5 + 3 * 3, -4.8315 + 1.26, 91], [75.5 + 3 * 3, -4.8315 - 1.26, 92],
         [75.5 + 3 * 4, -4.8315 + 1.26, 101], [75.5 + 3 * 4, -4.8315 - 1.26, 102],
         [99 + 3 * 0, -8.0525 + 1.5, 111], [99 + 3 * 0, -8.0525 - 1.5, 112],
         [99 + 3 * 1, -8.0525 + 1.5, 111], [99 + 3 * 1, -8.0525 - 1.5, 112],
         [99 + 3 * 2, -8.0525 + 1.5, 111], [99 + 3 * 2, -8.0525 - 1.5, 112],
         [99 + 3 * 3, -8.0525 + 1.5, 111], [99 + 3 * 3, -8.0525 - 1.5, 112],
         [99 + 3 * 4, -8.0525 + 1.5, 111], [99 + 3 * 4, -8.0525 - 1.5, 112]
]
car_width = 1.568
car_length = 4.3
cones_x = np.array(cones)[:,0]
cones_y = np.array(cones)[:,1]

def check_collision(car_x, car_y, car_yaw, car_width, car_length, cone_x, cone_y):
    # 차량의 중심 좌표 계산
    cx = car_x - car_length / 2 * np.cos(car_yaw)
    cy = car_y - car_length / 2 * np.sin(car_yaw)

    # 콘의 좌표를 차량 중심을 기준으로 변환
    dx = cone_x - cx
    dy = cone_y - cy

    # 차량의 회전각을 사용하여 콘의 좌표를 변환
    rotated_dx = dx * np.cos(-car_yaw) - dy * np.sin(-car_yaw)
    rotated_dy = dx * np.sin(-car_yaw) + dy * np.cos(-car_yaw)

    # 충돌 검사
    if (abs(rotated_dx) <= car_length / 2) and (abs(rotated_dy) <= car_width / 2):
        return True
    else:
        return False


collision_ipg_coords = set()
collision_rl_coords = set()

collision_coords = {name: set() for name in datasets}

for name, data in zip(datasets, data_dicts):
    car_x_values = data['carx']
    car_y_values = data['cary']
    car_yaw_values = data['caryaw']

    for i in range(len(car_x_values)):
        for j in range(len(cones_x)):
            if check_collision(car_x_values[i], car_y_values[i], car_yaw_values[i], car_width, car_length, cones_x[j], cones_y[j]):
                collision_coords[name].add((cones_x[j], cones_y[j]))

# Print collisions for each dataset
for name, coords in collision_coords.items():
    print(f"{name.upper()} collisions coordinates: {coords}")


def plot_car_trajectory(cones, data, traj_tx, traj_ty, label, collision_cones=None):
    car_width = 1.568
    carx = data['carx']
    cary = data['cary']
    caryaw = data['caryaw']
    cones_x = np.array(cones)[:, 0]
    cones_y = np.array(cones)[:, 1]

    plt.scatter(cones_x, cones_y, marker='s', label='Cone', color='orange')
    if collision_cones:
        collision_x = [x[0] for x in collision_cones]
        collision_y = [x[1] for x in collision_cones]
        plt.scatter(collision_x, collision_y, marker='s', label='Collision Cone', color='red')

    carx_upper = carx + np.sin(caryaw) * car_width / 2
    carx_lower = carx - np.sin(caryaw) * car_width / 2
    cary_upper = cary + np.cos(caryaw) * car_width / 2
    cary_lower = cary - np.cos(caryaw) * car_width / 2

    plt.plot(carx_upper, cary_upper, label=label, color='blue')
    plt.plot(carx_lower, cary_lower, color='blue')

    plt.plot(traj_tx, traj_ty, label='Trajectory', color='green')
    plt.xlabel("Distance (m)")
    plt.ylabel("Distance (m)")
    plt.xlim([0, 161])
    plt.ylim([-15, 0])
    plt.legend()
    plt.show()


for dataset, data_dict in zip(datasets, data_dicts):
    plot_car_trajectory(cones, data_dict, traj_tx, traj_ty, dataset.upper(), collision_coords[dataset])

for dataset, data_dict in zip(datasets, data_dicts):
    plt.plot(data_dict['carx'], data_dict['cary'], label=dataset.upper())
plt.scatter(cones_x, cones_y, marker='s', label='Cone', color='orange')
plt.plot(traj_tx, traj_ty, label='Trajectory', color='green')
plt.legend()
plt.xlabel("Distance (m)")
plt.ylabel("Distance (m)")
plt.show()
