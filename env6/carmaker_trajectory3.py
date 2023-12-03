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
            [carx, cary, caryaw],
            [6, 6, 6, action1, action2]
        )
        self.xy = self.b.get_xy_points()
        self.end_point = self.b.get_xy_point(1)

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
            arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj.csv").loc[:, ["traj_tx", "traj_ty"]].values
            return calculate_dev_low([carx, cary, caryaw], arr)
        else:
            return self.calculate_dev_b(carx, cary, caryaw)

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
    road_type, low = "DLC", True
    traj = Trajectory(road_type=road_type, low=low)
    traj.plot()

