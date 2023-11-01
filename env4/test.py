import bezier
import numpy as np
import matplotlib.pyplot as plt

# 제어점 설정
nodes = np.asfortranarray([
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # x 좌표
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # y 좌표
])

# 베지에 커브 객체 생성
curve = bezier.Curve(nodes, degree=6)
print(curve)
# 커브 플로팅
fig, ax = plt.subplots()
curve.plot(num_pts=256, ax=ax)

# 제어점 플로팅
ax.plot(nodes[0, :], nodes[1, :], linestyle="None", marker="x", color="black")

plt.show()
