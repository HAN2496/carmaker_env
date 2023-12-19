#!/usr/bin/env python
# coding: utf-8

# python == 3.7  
# bezier == 2021.2.12  
# numpy == 1.21.6  
# matplotlib == 3.5.2

import bezier
import numpy as np
import matplotlib.pyplot as plt

class Bezier:
    def __init__(self, dt, N):

        self.dt = dt # 간격
        self.N = N # 개수
        self.update(
            np.zeros((3,)),
            [1, 1, 1, 0.0, 0.0]
        )


    def update(self, x0, action):
        '''
        Parameters
        ----------
        x0: A starting pose [X0, Y0, psi0]^T.
        v0: A starting linear velocity.
        action: Bezier curve parameters
            action[0]: Length between node 1 and node 0. = 6
            action[1]: Length between node 2 and node 1. = 6
            action[2]: Length between node 3 and node 2. = 6
            action[3]: Angle difference between edge 01 and edge 12.
            action[4]: Angle difference between edge 01 and edge 23.
        '''

        self.x0 = x0

        # Bezier curve for position profile: p(s)
        self.p = [np.array(x0[:2])]
        for l, a in zip(action[:3], x0[-1] + np.r_[0, action[3:5]]):
            self.p.append(self.p[-1] + l * np.array([np.cos(a), np.sin(a)]))
        self.p_curve = bezier.Curve(np.asfortranarray(self.p).T, degree=3)

    def get_xy_points(self):
        t = np.arange(0, 1+self.dt, self.dt)
        points = self.p_curve.evaluate_multi(t)
        return points.T
    def get_ctrl_points(self):
        return self.p

    def get_ctrl_last_point(self):
        return np.array(self.get_ctrl_points())[-1, :]
    def show_curve(self):
        t = np.arange(0, 1+self.dt, self.dt)
        points = self.p_curve.evaluate_multi(t)
        plt.plot(*points)
        ctrl_points = np.array(self.get_ctrl_points())
        plt.scatter(ctrl_points[:, 0], ctrl_points[:, 1], color='red',  facecolors='none')
        plt.plot(ctrl_points[:, 0], ctrl_points[:, 1], color='orange')
        for x, y in ctrl_points:
            plt.text(x, y - 0.3, f'({x:.2f}, {y:.2f})', ha='center', va='bottom')

        plt.title("Bezier Curve And Control Points")
        plt.show()

if __name__ == '__main__':

    B = Bezier(0.01, 40)
    B.update(
        np.zeros((3,)),
        [6, 6, 6, np.pi/2, np.pi/3]
    )

    B.show_curve()
    print(B.get_ctrl_last_point())
#    print(B.get_xy_points())
