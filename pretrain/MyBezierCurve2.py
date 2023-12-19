
# python == 3.7
# bezier == 2021.2.12
# numpy == 1.21.6
# matplotlib == 3.5.2

import bezier
import numpy as np
import matplotlib.pyplot as plt

class BezierReference:
    def __init__(self, x0, dt=0.05):

        self.dt = dt  # 간격
        self.update(
            x0,
            [6, 6, 6, 0, 0]
        )

    def update(self, x0, action):
        '''
        x0: A starting pose [X0, Y0, psi0]^T.
        action: Bezier curve parameters
            action[0]: Length between node 1 and node 0.
            action[1]: Length between node 2 and node 1.
            action[2]: Length between node 3 and node 2.
            action[3]: Angle difference between edge 01 and edge 12.
            action[4]: Angle difference between edge 01 and edge 23.
        '''

        self.x0 = x0

        # Bezier curve for position profile: p(s)
        self.p = [np.array(x0[:2])]
        for l, a in zip(action[:3], x0[-1] + np.r_[0, action[3:5]]):
            self.p.append(self.p[-1] + l * np.array([np.cos(a), np.sin(a)]))
        self.p_curve = bezier.Curve(np.asfortranarray(self.p).T, degree=3)
        self.p = np.array(self.p)

    def update_with_endpoints(self, start_data, end_data, scale=1):
        """
        start_data: 시작점 데이터 [x, y, angle] - 위치와 기울기 각도 (라디안).
        end_data: 끝점 데이터 [x, y, angle] - 위치와 기울기 각도 (라디안).
        scale: 제어점 오프셋 스케일.
        """
        # 시작점과 끝점 위치 추출
        p0 = np.array(start_data[:2])
        p3 = np.array(end_data[:2])

        # 시작점과 끝점의 기울기 각도 추출
        start_angle = start_data[2]
        end_angle = end_data[2]

        # 제어점 계산
        p1 = p0 + scale * np.array([np.cos(start_angle), np.sin(start_angle)])
        p2 = p3 - scale * np.array([np.cos(end_angle), np.sin(end_angle)])

        # 제어점으로 베지어 곡선 생성
        self.p = [p0, p1, p2, p3]
        self.p = np.array(self.p)
        self.p_curve = bezier.Curve(np.asfortranarray([p0, p1, p2, p3]).T, degree=3)

    def get_ctrl_points(self):
        return self.p

    def get_curve(self):
        return self.p_curve

    def get_xy_point(self, t):
        t = np.array([t], dtype=float)
        points = self.p_curve.evaluate_multi(t).T
        return points.flatten()

    def get_xy_points(self):
        t_values = np.arange(0, 1 + self.dt, self.dt)
        points = self.p_curve.evaluate_multi(t_values).T
        return np.array(points)
    def show_curve(self, num_points=100):
        points = self.p_curve.evaluate_multi(np.linspace(0.0, 1.0, num_points))
        plt.scatter(points[0, :], points[1, :], color='red')

        ctrl_points = np.array(self.p_curve.nodes).T
        plt.scatter(ctrl_points[:, 0], ctrl_points[:, 1], color='red', facecolors='none')
        for x, y in ctrl_points:
            plt.text(x + 0.1, y + 0.1, f'({x:.2f}, {y:.2f})', ha='center', va='bottom')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("3rd degree Bezier reference curve ")
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    B = BezierReference([0,0,0], 0.01)

    B.update(
        [0, 0, 0],  # x0: 시작 위치
        [6, 6, 6, np.pi/3, np.pi/2]  # action: 제어점 설정
    )
    print(B.get_xy_point(1))

    B.show_curve()
    """
    B.update_with_endpoints(
        [0, 0, np.pi/3],
        [12, 0, np.pi/3]
    )
    B.show_curve()
    """