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
    def update_traj(self, carx, action):
        action = action[0] / np.pi / 12
        self.b.add_curve(
            [1, 1, 1, action]
        )
        self.arr = np.concatenate((self.arr, self.b.get_last_xy_points()), axis=0)

    def get_last_traj_x(self):
        self.last_traj_x_dist = self.b.curves[-1].nodes[0, -1]
        return self.b.curves[-1].nodes[0, -1]
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
        if self.low:
            return self.calculate_dev_low(carx, cary, caryaw)
        else:
            return self.calculate_dev_b(carx, cary, caryaw)

    def calculate_dev_low(self, carx, cary, caryaw):
        arr = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:, ["traj_tx", "traj_ty"]].values
        """
        if self.road_type == "DLC":
            return self.calculate_dev_DLC(carx, cary, caryaw)
        elif self.road_type == "SLALOM2":
            return self.calculate_dev_SLALOM2(carx, cary, caryaw)
        """
        return calculate_dev([carx, cary, caryaw], arr)

    def calculate_dev_b(self, carx, cary, caryaw):
        arr = self.b.get_xy_points(carx)
        return calculate_dev([carx, cary, caryaw], arr)

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
    traj.update_traj(10, 1)
    print(traj.get_last_traj_x())
    traj.update_traj(20, -1)
    print(traj.get_last_traj_x())
    traj.update_traj(30, -1)
    print(traj.get_last_traj_x())
    traj.plot()
