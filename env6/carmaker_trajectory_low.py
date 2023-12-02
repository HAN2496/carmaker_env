from MyBezierCurve import BezierCurve
from carmaker_cone import *
from common_functions import *

class Trajectory:
    def __init__(self, road_type, low=True):
        self.road_type = road_type
        self.low = low
        self._init_traj()
        self.end_point = []
    def _init_traj(self):
        if self.low:
            self.arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj.csv").loc[:,
                             ["traj_tx", "traj_ty"]].values
        else:
            x, y = init_car_pos(self.road_type)
            self.b = BezierCurve(x, y, 0.02)
            self.arr = self.b.get_all_xy_points()
            self.end_point = self.b.get_last_point()

    def find_lookahead_traj(self, x, y, distances):
        distances = np.array(distances)
        result_points = []

        min_idx = np.argmin(np.sum((self.arr - np.array([x, y])) ** 2, axis=1))

        for dist in distances:
            lookahead_idx = min_idx
            total_distance = 0.0
            while total_distance < dist and lookahead_idx + 1 < len(self.arr):
                total_distance += np.linalg.norm(self.arr[lookahead_idx + 1] - self.arr[lookahead_idx])
                lookahead_idx += 1

            if lookahead_idx < len(self.arr):
                result_points.append(self.arr[lookahead_idx])
            else:
                result_points.append(self.arr[-1])

        return np.array(result_points)
    def find_traj_points(self, carx, distances):
        points = []
        for distance in distances:
            x_diff = np.abs(self.arr[:, 0] - (carx + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(self.arr[nearest_idx])
        return np.array(points)

    def calculate_dev(self, carx, cary, caryaw):
        arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:, ["traj_tx", "traj_ty"]].values
        return calculate_dev([carx, cary, caryaw], arr)

    def plot(self, show=True):
        for idx, (x, y) in enumerate(self.arr):
            if idx == 0:
                plt.scatter(x, y, s=5, color='red', label='Trajectory')
            elif idx % 50 == 0 and self.low:
                plt.scatter(x, y, s=5, color='red')
            elif not self.low:
                plt.scatter(x, y, s=5, color='red')
        if show:
            plt.legend()
            plt.show()

if __name__ == "__main__":
    road_type, low = "DLC", False
    traj = Trajectory(road_type=road_type, low=low)

