import matplotlib.pyplot as plt
import numpy as np

# 1. y = -5.25인 직선
x = np.linspace(0, 400, 1000)
y = [-5.25 for _ in x]

plt.plot(x, y, label='y=-5.25', color='blue')

# 2. (100, -5.25)를 기준으로 x값을 30만큼 증가시킨 100 개의 콘.
x_cones = [100 + i*30 for i in range(10)]
y_cones = [-5.25 for _ in x_cones]
plt.scatter(x_cones, y_cones, label='Cones', color='red')

# 3. (10, -5.25)을 중심으로 하는 직사각형
rectangle = plt.Rectangle((5, -5.25), 4.3, 1.5, fill=True, color='green', label='car')
plt.gca().add_patch(rectangle)

# 그래프 출력
plt.xlim(0, 400)
plt.ylim(-10, 0)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.show()
