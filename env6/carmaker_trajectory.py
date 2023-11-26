from MyBezierCurve import BezierCurve
from carmaker_cone import *

class Trajectory:
    def __init__(self, road_type, low=True):
        self.road_type = road_type
        self.low = low

    def find_lookahead_traj(self, x, y, yaw, distances):
        distances = np.array(distances)
        result_points = []

        min_idx = np.argmin(np.sum((self.traj_data - np.array([x, y])) ** 2, axis=1))

        for dist in distances:
            lookahead_idx = min_idx
            total_distance = 0.0
            while total_distance < dist and lookahead_idx + 1 < len(self.traj_data):
                total_distance += np.linalg.norm(self.traj_data[lookahead_idx + 1] - self.traj_data[lookahead_idx])
                lookahead_idx += 1

            if lookahead_idx < len(self.traj_data):
                result_points.append(self.traj_data[lookahead_idx])
            else:
                result_points.append(self.traj_data[-1])

        return np.array(result_points)

if __name__ == "__main__":
    road_type = "UTurn"
