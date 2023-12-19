import bezier
import numpy as np
import matplotlib.pyplot as plt

"""
action에 의해 넣을때는 add_curve만 쓰면 됨 ㅇㅇ.
"""
class BezierCurve:
    def __init__(self, carx, cary, dt):
        self.dt = dt
        self.curves = []  # 여러 베지어 곡선을 저장할 리스트
        self.update(
            [carx - 1, cary, 0],
            [6, 6, 6, 0.0, 0.0]
        )
        self.last_angle = 0

    def update(self, x0, action):
        action = action
        new_p = [np.array(x0[:2])]

        angles = x0[-1] + np.cumsum([0] + action[3:5])

        for l, a in zip(action[:3], angles):
            new_p.append(new_p[-1] + l * np.array([np.cos(a), np.sin(a)]))

        new_curve = bezier.Curve(np.asfortranarray(new_p).T, degree=3)
        self.curves.append(new_curve)
        self.last_angle = angles[-1]  # 마지막 각도 업데이트

    def add_curve(self, action):
        x0 = np.array(self.curves[-1].nodes[:,-1])  # 마지막 곡선의 마지막 컨트롤 포인트
        x0 = np.append(x0, self.last_angle)  # psi0 값 (임시로 0)
        action = action + [0]
        self.update(x0, action)

    def get_last_point(self):
        last_curve = self.curves[-1]
        last_point = last_curve.nodes[:, -1]
        return last_point

    def get_last_xy_points(self):
        t = np.arange(0, 1 + self.dt, self.dt)
        points = []
        curve_points = self.curves[-1].evaluate_multi(t).T
        points.append(curve_points)
        concatenated_points = np.vstack(points)
        return np.array(concatenated_points)

    def get_all_xy_points(self):
        t = np.arange(0, 1, self.dt)
        points = []
        for curve in self.curves:
            curve_points = curve.evaluate_multi(t).T
            points.append(curve_points)
        concatenated_points = np.vstack(points)
        return np.array(concatenated_points)

    def get_xy_points(self, x):
        t = np.arange(0, 1, self.dt)
        points = []
        check_curves= []
        for curve in self.curves:
            first_point = curve.nodes[0, 0]
            last_point = curve.nodes[0, -1]
            if first_point < x < last_point or last_point < x < first_point:
                check_curves.append(curve)
        if len(check_curves) == 0:
            check_curves = self.curves
        for curve in check_curves:
            curve_points = curve.evaluate_multi(t).T
            points.append(curve_points)
        concatenated_points = np.vstack(points)
        return np.array(concatenated_points)

    def show_curve(self):
        for curve in self.curves:
            t = np.arange(0, 1+self.dt, self.dt)
            points = curve.evaluate_multi(t)
            plt.plot(*points)

            # 각 곡선의 컨트롤 포인트 시각화
            ctrl_points = np.array(curve.nodes).T
            plt.scatter(ctrl_points[:, 0], ctrl_points[:, 1], color='red', facecolors='none')
#            plt.plot(ctrl_points[:, 0], ctrl_points[:, 1], color='orange')
            for x, y in ctrl_points:
                plt.text(x + 0.1, y + 0.1, f'({x:.2f}, {y:.2f})', ha='center', va='bottom')

        plt.title("Bezier Curve And Control Points")
        plt.axis('equal')
        plt.show()
# 사용 예시
if __name__ == '__main__':
    B = BezierCurve(1, 0, 0.001)
    B.show_curve()
    B.add_curve([1, 1, 1, 0])
    B.show_curve()
    B.add_curve([5, 5, 5, np.pi/3])
    #B.show_curve()
    B.add_curve([5, 5, 5, np.pi/6])
    print(B.get_xy_points(5))
    B.show_curve()

