from numpy import genfromtxt
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def erase_similar_x(data, epsilon=0.009):

    tmp = data[0][0]
    i = 1
    while True:
        if data[i][0] == 500.0:
            break
        if np.abs(data[i][0] - tmp) <= epsilon:
            print("found", i)
            data = np.delete(data, i, axis=0)
        else:
            tmp = data[i][0]
            i +=1
    return data

def skip_steps(data, step = 4):
    return data[::step]

def calculate_theta(spline, data):
    theta = []
    for i in range(data.shape[0]):
        length = quad(
            lambda x: np.sqrt(1 + spline(x)**2),
            0,
            data[i][0]
        )
        print(i, length)
        theta.append(length)
        if length < theta[-1]:
            print("Error: Calculating gone wrong.")
            break
    
    return np.array(theta)

def calculate_psi(spline, data):
    return np.arctan2(spline(data[:, 1], 1), 1)


if __name__=="__main__":
    track_file = "traj_slalom.csv"
    data = genfromtxt(track_file, delimiter=',', skip_header = 1)
    data = data[8500:] #salom starts at 8500

    data = erase_similar_x(data)
    data = skip_steps(data)
    print(data.shape)


    cs = CubicSpline(data[:, 0], data[:, 1])

    psi = calculate_psi(cs, data)
    print(data[0], data[1], psi[0])
    print(psi.shape)
    # plt.figure(1)
    # plt.plot(data[:, 0], data[:, 1])
    # plt.plot(data[:,0], cs(data[:, 0]))
    # plt.legend(["data", "cubic spline"])
    # plt.show()



# print(theta)
# print(theta.shape)