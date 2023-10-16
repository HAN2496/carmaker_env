import numpy as np
from scipy.interpolate import interp1d

class Trajectory:
    def __init__(self, first_x=2.1976, first_y=-10, point_interval=2, point_num=5, arr_interval=0.01):
        self.x0 = first_x
        self.y0 = first_y
        self.point_interval=point_interval
        self.point_num=point_num
        self.sight = self.point_interval * (self.point_num - 1)
        self.arr_interval=arr_interval

        self.traj_point = np.array([self.x0 + self.sight, self.y0])
        self.traj_arr = self.init_traj_arr()

    def make_traj_point(self, x, y, action):
        new_traj_point = np.array([x + self.sight, y + action * 2])
        self.traj_point = new_traj_point
        return new_traj_point

    def init_traj_arr(self, traj_point):
        x1 = self.traj_point[0]
        y1 = self.traj_point[1]
        linear_interp = interp1d(np.array([self.x0, x1]), np.array([self.y0, y1]))
        xnew = np.arange(self.x0, x1, self.arr_interval)
        ynew = linear_interp(xnew)
        return np.array(list(zip(xnew, ynew)))
    def make_trajectory(self, traj_point):
        if traj_point[0] <= self.traj_arr[-1]