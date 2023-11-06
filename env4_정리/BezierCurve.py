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
        self.dt = dt
        self.N = N

class BezierReference:
    
    '''
    This class is for creating a curve using one 3rd degree bezier curve in 3D space.
    '''
    
    def __init__(self, dt, N):

        self.dt = dt # 간격
        self.N = N # 개수
        self.update(
            np.zeros((3,)), 0.0,
            [1, 1, 1, 0.0, 0.0, 0.5]
        )


    def update(self, x0, v0, action):
        '''
        Parameters
        ----------
        x0: A starting pose [X0, Y0, psi0]^T.
        v0: A starting linear velocity.
        action: Bezier curve parameters
            action[0]: Length between node 1 and node 0.
            action[1]: Length between node 2 and node 1.
            action[2]: Length between node 3 and node 2.
            action[3]: Angle difference between edge 01 and edge 12.
            action[4]: Angle difference between edge 01 and edge 23.
            action[5]: A parameter for the time profile curve, bounded by [0, 1].
        '''

        self.x0 = x0
        self.v0 = v0

        # Bezier curve for position profile: p(s)
        self.p = [np.array(x0[:2])]
        for l, a in zip(action[:3], x0[-1] + np.r_[0, action[3:5]]):
            self.p.append(self.p[-1] + l * np.array([np.cos(a), np.sin(a)]))
        self.p_curve = bezier.Curve(np.asfortranarray(self.p).T, degree=3)

        # Bezier curve for curve parameter profile given normalized time: s(t)
        dpds = self.p_curve.evaluate_hodograph(0.0).flatten()  # dp/ds
        dsdt = v0 / np.linalg.norm(dpds)  # ds/dt = (dp/dt) * (ds/dp)
        dsdnt = dsdt * self.N * self.dt
        self.s = [0.0, dsdnt / 3.0, action[-1], 1.0] # control points
        self.s_curve = bezier.Curve(np.asfortranarray([self.s]), degree=3)


    def get_ctrl_points(self): 
        return self.p, self.s

    def get_curve(self):
        return self.p_curve, self.s_curve

    def normalize_t(self, t):
        return t / (self.dt * self.N)

    def get_s(self, t):
        return self.s_curve.evaluate(
            self.normalize_t(t)
        )[0, 0]

    def get_xt(self, t):
        s = self.get_s(t)
        dpds = self.p_curve.evaluate_hodograph(s).flatten()
        return np.r_[
            self.p_curve.evaluate(s).flatten(),  # X_k, Y_k
            np.arctan2(dpds[1], dpds[0])             # psi_k
        ]

    def get_vt(self, t):
        s = self.get_s(t)
        nt = self.normalize_t(t)
        dpds = self.p_curve.evaluate_hodograph(s).flatten()
        dsdt = self.s_curve.evaluate_hodograph(nt)[0, 0] / (self.N * self.dt)
        return np.linalg.norm(dpds) * dsdt  # v_k

    def get_reference(self):

        ref = [np.r_[self.x0, self.v0]]

        for k in range(1, self.N+1):
            nt = k / self.N
            s = self.s_curve.evaluate(nt)[0, 0]
            dpds = self.p_curve.evaluate_hodograph(s).flatten()
            dsdt = self.s_curve.evaluate_hodograph(nt)[0, 0] / (self.N * self.dt)
            ref.append(
                np.r_[
                    self.p_curve.evaluate(s).flatten(),  # X_k, Y_k
                    np.arctan2(dpds[1], dpds[0]),            # psi_k
                    np.linalg.norm(dpds) * dsdt          # v_k
                ]
            )

        return np.array(ref)
        
    def show_curve(self, figsize=(10, 10), num_points=100, color = 'r'):
        
        fig1 = plt.figure(figsize = figsize)
        points = self.p_curve.evaluate_multi(np.linspace(0.0, 1.0, num_points))
        plt.plot(points[0, :], points[1, :], color)
        ref = self.get_reference()
        plt.scatter(ref[:, 0], ref[:, 1], c=ref[:, -1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("3rd degree Bezier reference curve ")
        plt.axis('equal')
        fig2 = plt.figure(figsize = figsize)
        plt.plot(np.linspace(0,self.N*self.dt, self.N+1), ref[:, -1], color)
        plt.xlabel("t")
        plt.ylabel("v(t)")
        plt.title("linear velocity profile")
        plt.show()


    
if __name__ == '__main__':

    B = BezierReference(0.01, 40)
    B.update(
        np.zeros((3,)), 3.0,
        [6, 6, 6, np.pi/2, 0.0, 1.0]
    )

    print(B.get_vt(0.0))
    B.show_curve()
