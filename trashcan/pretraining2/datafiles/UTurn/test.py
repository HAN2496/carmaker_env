import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def rotate_point(x, y, x0, y0, a):
    # 각도를 라디안으로 변환
    radians = np.radians(a)

    # (x, y)를 원점으로 이동
    x_shifted = x - x0
    y_shifted = y - y0

    # 회전 변환 수행
    x_rotated = x_shifted * np.cos(radians) - y_shifted * np.sin(radians)
    y_rotated = x_shifted * np.sin(radians) + y_shifted * np.cos(radians)

    # 회전된 좌표를 원래 위치로 이동
    x_final = x_rotated + x0
    y_final = y_rotated + y0

    return np.array([x_final, y_final])


df = pd.read_csv('test.csv').loc[:, ['traj_tx', 'traj_ty']].values

data = []

for x, y in df:
    data.append(rotate_point(x, y, 0, 0, 270))

data = np.array(data)
df = pd.DataFrame(data, columns=['traj_tx', 'traj_ty'])

# DataFrame을 CSV 파일로 저장
df.to_csv('output.csv', index=False)

plt.scatter(data[:, 0], data[:, 1])

plt.show()