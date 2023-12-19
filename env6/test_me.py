import numpy as np

# 예시 데이터: m x 2 형태의 넘파이 배열
data = np.array([[1, 3],
                 [2, 4],
                 [5, 8],
                 [7, 9]])

a = [1, 1]
distances = np.sqrt((data[:, 0] - a[0])**2 + (data[:, 1] - a[1])**2)
xy_diff = np.linalg.norm(data -a, axis=1)
xy_diff2 = np.diff(data, axis=0)
xy_diff3 = np.linalg.norm(xy_diff2 -np.array([0, 0]), axis=1)
#print(data[0])
#print(xy_diff)
#print(distances)
print(xy_diff2)
print(xy_diff3)
