from MyBezierCurve2 import BezierReference as BezierCurve
from carmaker_cone import *
from common_functions import *
import math
import bezier
class Trajectory:
    def __init__(self, road_type, low=True):
        self.road_type = road_type
        self.low = low
        self.start_point = []
        self.end_point = []
        self._init_traj()

    def _init_traj(self):
        print('here')
        if self.low:
            self.xy = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj.csv").loc[:,
                             ["traj_tx", "traj_ty"]].values
        else:
            x, y = init_car_pos(self.road_type)
            self.b = BezierCurve([x, y, 0], dt=0.02)
            _, _, p0, p1 = self.b.get_ctrl_points()
            self.start_point = [p0, p1]
            self.end_point = self.b.get_xy_point(1)
            self.xy = self.b.get_xy_points()
    def update_traj(self, car_pos, action):
        #print("Update")
        carx, cary, caryaw = car_pos
        action1 = action[0] #/ np.pi
        action2 = action[1] #/ np.pi
        self.b.update(
            [carx + 12, cary, caryaw],
            [1, 1, 1, action1, action2]
        )
        self.connect_traj()
        _, _, p0, p1 = self.b.get_ctrl_points()
        self.start_point = [p0, p1]
        self.xy = np.concatenate((self.xy, self.b.get_xy_points()), axis=0)
        self.end_point = self.b.get_xy_point(1)
    def connect_traj(self):
        p0, p1 = self.start_point[0], self.start_point[1]
        p2, p3, _, _ = self.b.get_ctrl_points()
        start_point_angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        end_point_angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])
        p_connect = BezierCurve([0, 0, 0])
        p_connect.update_with_endpoints(
            [p1[0], p1[1], start_point_angle],
            [p2[0], p2[1], end_point_angle]
        )
        self.b_before = p_connect
        self.xy = np.concatenate((self.xy, p_connect.get_xy_points()), axis=0)
        #p_connect.show_curve()

    def find_lookahead_traj(self, x, y, distances):
        distances = np.array(distances)
        result_points = []

        min_idx = np.argmin(np.sum((self.xy - np.array([x, y])) ** 2, axis=1))

        for dist in distances:
            lookahead_idx = min_idx
            total_distance = 0.0
            while total_distance < dist and lookahead_idx + 1 < len(self.xy):
                total_distance += np.linalg.norm(self.xy[lookahead_idx + 1] - self.xy[lookahead_idx])
                lookahead_idx += 1

            if lookahead_idx < len(self.xy):
                result_points.append(self.xy[lookahead_idx])
            else:
                result_points.append(self.xy[-1])

        return np.array(result_points)

    def find_traj_points(self, carx, distances):
        points = []
        for distance in distances:
            x_diff = np.abs(self.xy[:, 0] - (carx + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(self.xy[nearest_idx])
        return np.array(points)

    def calculate_dev(self, carx, cary, caryaw):
        if self.low:
            return self.calculate_dev_low(carx, cary, caryaw)
        else:
            return self.calculate_dev_b(carx, cary, caryaw)

    def calculate_dev_low(self, carx, cary, caryaw):
        if self.road_type == "DLC":
            return self.calculate_dev_DLC(carx, cary, caryaw)
        elif self.road_type == "SLALOM2":
            return self.calculate_dev_SLALOM2(carx, cary, caryaw)

    def calculate_dev_b(self, carx, cary, caryaw):
        pass

    def calculate_dev_DLC(self, carx, cary, caryaw):
        if carx <= 62:
            return np.array([cary + 10, caryaw])
        elif 75.5 <= carx <= 86.5:
            return np.array([cary + 6.485, caryaw])
        elif 99 <= carx:
            return np.array([cary + 10.385, caryaw])
        else:
            arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:, ["traj_tx", "traj_ty"]].values
            return calculate_dev([carx, cary, caryaw], arr)

    def calculate_dev_SLALOM2(self, carx, cary, caryaw):
        if carx <= 85:
            return np.array([cary + 25, caryaw])
        elif carx >= 400:
            return np.array([cary + 25, caryaw])
        else:
            arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:, ["traj_tx", "traj_ty"]].values
            return calculate_dev([carx, cary, caryaw], arr)

    def calculate_dev_crc(self, carx, cary, caryaw):
        if carx <= 100:
            return np.array([cary, caryaw])
        else:
            norm_yaw = np.mod(caryaw, 2 * np.pi)
            devDist = 30 - np.linalg.norm(np.array([carx, cary]) - np.array([100, 30]))
            devAng = norm_yaw - np.mod(np.arctan2(cary - 30, carx - 100) + np.pi / 2, 2 * np.pi)
            devAng = (devAng + np.pi) % (2 * np.pi) - np.pi
            return np.array([devDist, devAng])

    def save(self):
        pass
    def plot(self, show=True):
        max_idx = len(self.xy) - 1
        for idx, (x, y) in enumerate(self.xy):
            if not self.low:
                # Fading from deep red to lighter red
                color = np.array([1, 0, 0]) * (1 - idx / max_idx) + np.array([1, 0.5, 0.5]) * (idx / max_idx)
            else:
                color = 'red'
            if idx == 0:
                plt.scatter(x, y, s=5, color='red', label='Trajectory')
            elif idx % 50 == 0 and self.low:
                plt.scatter(x, y, s=5, color='red')
            elif not self.low:
                plt.scatter(x, y, s=5, color=color)

        if show:
            plt.legend()
            plt.axis('equal')
            plt.show()

if __name__ == "__main__":
    road_type, low = "DLC", False
    traj = Trajectory(road_type=road_type, low=low)
    traj.plot()
    traj.update_traj(
        [3, -10, 0],
        [1, -1]
    )
    #traj.plot()
    traj.update_traj(
        [5, -10, 0],
        [-1, 1]
    )
    #traj.plot()

    traj.update_traj(
        [6, -10, 0],
        [1, 1]
    )
    traj.plot()
    traj._init_traj()
    traj.plot()
