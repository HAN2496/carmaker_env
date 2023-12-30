from MyBezierCurve2 import BezierReference as BezierCurve
from carmaker_cone import *
from common_functions import *
import math
import bezier
class Trajectory:
    def __init__(self, road_type, distances, env_num=2, low=True):
        self.road_type = road_type
        self.section = 0
        self.distances = distances
        self.low = low
        self.env_num = env_num
        self.start_point = []
        self.end_point = []
        self._init_traj()
        self.devDist, self.devAng = 0, 0
        x, y, _ = init_car_pos(road_type=self.road_type)
        self.manage_traj([x, y, 0.681124, 0])

    def _init_traj(self):
        if self.env_num == 0:
            print(f'Init trajectory, Section changes to {self.section}')
        self.section = 0
        if self.low:
            if self.road_type == "Ramp":
                self.xy = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:,
                          ["traj_tx", "traj_ty"]].values
            else:
                self.xy = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj.csv").loc[:,
                                 ["traj_tx", "traj_ty"]].values
        else:
            x, y = init_car_pos(self.road_type)
            self.b = BezierCurve([x, y, 0], dt=0.02)
            _, _, p0, p1 = self.b.get_ctrl_points()
            self.start_point = [p0, p1]
            self.end_point = self.b.get_xy_point(1)
            self.xy = self.b.get_xy_points()

    def manage_traj(self, car_pos):
        carx, cary, carz, caryaw = car_pos
        self.find_lookahead_traj(carx, cary, caryaw)
        if 0 not in self.distances:
            self.devDist, self.devAng = self.calculate_dev(carx, cary, caryaw)
        if self.road_type == "Ramp" and self.low:
            self.is_traj_shoud_change(carx, cary, carz)

    def is_traj_shoud_change(self, carx, cary, carz):
        if self.section == 0:
            if carx >= 690:
                if self.env_num == 0:
                    print("Section change to 1")
                self.section = 1
                self.xy = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj2.csv").loc[:,
                          ["traj_tx", "traj_ty"]].values
        elif self.section == 1:
            if carx < 690 and cary > -100 and carz > -3:
                if self.env_num == 0:
                    print("Section change to 0")
                self.section = 0
                self.xy = pd.read_csv(f"datafiles/{self.road_type}/datasets_traj1.csv").loc[:,
                          ["traj_tx", "traj_ty"]].values
    def update_traj(self, car_pos, action):
        carx, cary, caryaw = car_pos
        action1 = action[0] #/ np.pi
        action2 = action[1] #/ np.pi
        self.b.update(
            [carx, cary, caryaw],
            [6, 6, 6, action1, action2]
        )
        self.xy = self.b.get_xy_points()
        self.end_point = self.b.get_xy_point(1)

    def find_lookahead_traj(self, x, y, yaw):

        result_points = []

        min_idx = np.argmin(np.sqrt(np.sum((self.xy - [x, y]) ** 2, axis=1)))

        for dist in self.distances:
            if dist == 0:
                self.devDist, self.devAng = calculate_dev_low([x, y, yaw], self.xy, index=min_idx)
            lookahead_idx = min_idx
            total_distance = 0.0
            while total_distance <= dist and lookahead_idx + 1 < len(self.xy):
                total_distance += np.linalg.norm(self.xy[lookahead_idx + 1] - self.xy[lookahead_idx])
                lookahead_idx += 1

            if lookahead_idx < len(self.xy):
                result_points.append(self.xy[lookahead_idx])
            else:
                result_points.append(self.xy[-1])


        self.lookahed_traj = np.array(result_points)

    def find_traj_points(self, x, y, yaw):
        points = []
        for distance in self.distances:
            x_diff = np.abs(self.xy[:, 0] - (x + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(self.xy[nearest_idx])
            if distance == 0:
                self.devDist, self.devAng = calculate_dev_low([x, y, yaw], self.xy, index=nearest_idx)
        self.lookahed_traj = np.array(points)

    def calculate_dev(self, carx, cary, caryaw):
        if self.low:
            return calculate_dev_low([carx, cary, caryaw], self.xy)
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

    def find_lookahead_traj_ramp(self, x, y, distances):
        pass

    def check_section_ramp(self, x, y, z):
        if x < 610:
            self.section = 0
        elif abs(x - 650.25) <= 5.25 and z <= -1:
            self.section = 2
        else:
            self.section = 1

    def calculate_dev_ramp(self, x, y, yaw):
        if self.section == 0:
            devDist = abs(y + 12.25)
            devAng = abs(yaw)
            return np.array([devDist, devAng])
        elif self.section == 2:
            devDist = abs(x - 650.25)
            devAng = abs(yaw + 4.71180055142883)
            return np.array([devDist, devAng])
        else:
            return calculate_dev_low([x, y, yaw], self.xy)

if __name__ == "__main__":
    road_type, low = "DLC", True
    traj = Trajectory(road_type=road_type, low=low)
    traj.plot()

