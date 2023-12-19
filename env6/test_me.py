import numpy as np

# 예시 데이터셋 생성
data = np.array([[1, 3], [2, 4], [3, 7], [4, 8]])

# 차이 계산
differences = data[:, 1] - data[:, 0]
print(differences)
# k번째와 k-1번째 요소 간 차이 계산
differences = np.diff(data, axis=0)
print(differences)