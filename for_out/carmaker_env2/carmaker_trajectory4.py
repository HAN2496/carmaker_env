from MyBezierCurve2 import BezierReference as BezierCurve
from carmaker_cone import *
from common_functions import *
import math
import bezier
class Trajectory:
    def __init__(self, road_type, carx, cary, distances, low=True):
        self.road_type = road_type
        self.section = 0
        self.low = low
        self.carx, self.cary = carx, cary
        self.distances = distances
        self.sight = distances[-1]
        self.start_point = []
        self.end_point = []
        self._init_traj()

    def _init_traj(self):
        if self.low:
            self.total_xy = pd.read_csv(f"datafiles/{self.road_type}/test.csv").loc[:,
                             ["traj_tx", "traj_ty"]].values
            self.total_xy_diff = np.linalg.norm(np.diff(self.total_xy, axis=0), axis=1)
            self.total_sign_diff = calculate_directions(self.total_xy)
            self.xy_diff = np.array([self.total_xy_diff[0]])
            self.xy = np.array([self.total_xy[0]])
            self.xy_idx = 0
            tmpx, tmpy = self.carx, self.cary
            self.carx, self.cary = self.total_xy[0]
            self.manage_traj([tmpx, tmpy], remove=False)
        else:
            x, y = init_car_pos(self.road_type)
            self.b = BezierCurve([x, y, 0], dt=0.02)
            _, _, p0, p1 = self.b.get_ctrl_points()
            self.start_point = [p0, p1]
            self.end_point = self.b.get_xy_point(1)
            self.xy = self.b.get_xy_points()

    def manage_traj(self, car_after, remove=True):
        car_afterx, car_aftery = car_after
        delta = np.sqrt((car_after[0] - self.carx) ** 2 + (car_after[1] - self.cary) ** 2)
        sign = calculate_directions([[self.carx, self.cary], [car_afterx, car_aftery]])
        diff = 0
        if sign * self.total_sign_diff[self.xy_idx] < 0:
            for _ in enumerate(self.total_xy):
                if self.xy_idx <= 0:
                    pass
                if diff <= delta + self.sight:
                    self.xy = np.vstack([self.total_xy[self.xy_idx], self.xy])
                    self.xy_diff = np.append(self.total_xy_diff[self.xy_idx], self.xy_diff)
                    if remove:
                        self.xy = self.xy[:-2]
                        self.xy_diff = self.xy_diff[:-2]
                    self.xy_idx -= 1
                    diff += self.total_xy_diff[self.xy_idx]
        elif sign * self.total_sign_diff[self.xy_idx] > 0:
            for _ in enumerate(self.total_xy):
                if diff <= delta:
                    self.xy = np.vstack([self.xy, self.total_xy[self.xy_idx]])
                    self.xy_diff = np.append(self.xy_diff, self.total_xy_diff[self.xy_idx])
                    if remove:
                        self.xy = self.xy[1:]
                        self.xy_diff = self.xy_diff[1:]
                    self.xy_idx += 1
                    diff += self.total_xy_diff[self.xy_idx]
        self.carx = car_after[0]
        self.cary = car_after[1]
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

    def find_traj_points(self, x, y, distances):
        if self.road_type == "Ramp":
            return self.find_lookahead_traj_ramp(x, y, distances)
        points = []
        for distance in distances:
            x_diff = np.abs(self.xy[:, 0] - (x + distance))
            nearest_idx = np.argmin(x_diff)
            points.append(self.xy[nearest_idx])
        return np.array(points)

    def calculate_dev(self, carx, cary, caryaw):
        if self.road_type == "Ramp":
            return self.calculate_dev_ramp(carx, cary, caryaw)
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
    road_type, low = "Ramp", True
    carx, cary = 0, 0
    distances = [0, 2, 4, 6, 8]
    traj = Trajectory(road_type=road_type, carx=carx, cary=cary, distances=distances, low=low)
    print(traj.xy)
    carx, cary = 5, 0
    traj.manage_traj([carx, cary])
    #print(traj.xy)
    #traj.plot()

