import matplotlib.pyplot as plt
import numpy as np

def cubic_interpolation(x0, y0, x1, y1):
    a = 2 * (y0 - y1) / (x1 - x0) ** 3
    b = -3 / 2 * a * (x1 - x0)
    d = y0

    def interpolator(x):
        return a * (x - x0) ** 3 + b * (x - x0) ** 2 + d

    return interpolator

def plot_cubic_interpolation(x0, y0, x1, y1):
    f = cubic_interpolation(x0, y0, x1, y1)
    x = np.linspace(x0, x1, 400)
    y = [f(xi) for xi in x]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'r-', label='Cubic Interpolation')
    plt.scatter([x0, x1], [y0, y1], color='blue', marker='o', label='Given Points')
    plt.title('Cubic Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# 사용 예시
x0, y0 = 0, 1
x1, y1 = 1, 2
plot_cubic_interpolation(x0, y0, x1, y1)
