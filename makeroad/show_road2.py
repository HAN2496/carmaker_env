import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
import matplotlib.pyplot as plt
import torch

class Cone:
    def __init__(self, x, y, offset, num_cones, gap, type):
        self.type = type
        self.positions = self._create_cones(x, y, offset, num_cones, gap, type)

    def _create_cones(self, x, y, offset, num_cones, gap, type):
        positions = []
        for _ in range(num_cones):
            if offset == 0:
                positions.append((x, y, type))
                x += gap
            else:
                positions.append((x, y - offset, type))
                positions.append((x, y + offset, type))
                x += gap
        return np.array(positions)

class Car:
    def __init__(self, x, y, yaw, sight=10):
        self.car_width = 1.568
        self.car_length = 4.3
        self.x = x
        self.y = y
        self.yaw = yaw
        self.sight = sight
        self.vehicle_polygon = Polygon([
            (self.x, self.y - self.car_width / 2),
            (self.x, self.y + self.car_width / 2),
            (self.x - self.car_length, self.y + self.car_width / 2),
            (self.x - self.car_length, self.y - self.car_width / 2)
        ])
        self.rotated_vehicle_polygon = rotate(self.vehicle_polygon, self.yaw, origin=(self.x, self.y), use_radians=True)

    def _rotate_coords(self, arr):
        rad = -self.yaw
        R = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        arr -= np.array([self.x, self.y])
        return np.dot(arr, R)

    def get_max_cones_in_sight(self, cone_gap):
        return int(self.sight * cone_gap * 2)

    def check_collision(self, cone):
        for cone_x, cone_y in cone:
            cone_polygon = Point(cone_x, cone_y)
            if self.rotated_vehicle_polygon.intersects(cone_polygon):
                return np.array([cone_x, cone_y])

        return np.array([0, 0])

    def get_cone_in_sight(self, cone, plot=False):
        cone_arr = []
        for cone_x, cone_y, cone_type in cone:
            rel_cone = self._rotate_coords(np.array([cone_x, cone_y]))
            if 0 <= rel_cone[0] <= self.sight:
                cone_arr.append([cone_x, cone_y])

        # Padding
        max_cones = int(self.get_max_cones_in_sight(3) / 2) # 3 is cone gap
        if len(cone_arr) < max_cones:
            padding = max_cones - len(cone_arr)
            for _ in range(padding):
                cone_arr.append([self.x, self.y])  # padding with 0s

        if np.array(cone_arr).size > 0:
            cone_arr_rel = self._rotate_coords(np.array(cone_arr))
        else:
            cone_arr_rel = np.empty((0,2))
            cone_arr = np.empty((0, 2))

        if plot:
            return cone_arr_rel

        return cone_arr_rel.flatten()


    def _plot_absolute(self, cone):
        # Plot car
        plt.plot(*self.rotated_vehicle_polygon.exterior.xy, color='black', label='Vehicle')
        plt.scatter(self.x, self.y, color='black')

        # Plot cones
        for cone_x, cone_y, cone_type in cone:
            if cone_type == 2:
                plt.scatter(cone_x, cone_y, color='red')
            else:
                plt.scatter(cone_x, cone_y, color='green')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.ylim(-14, 1)
        plt.show()

    def _plot_relative(self, cone):
        # Get cones in sight
        cones_in_sight = self.get_cone_in_sight(cone, plot=True)

        # Plot cones in sight
        for cone_x, cone_y in cones_in_sight:
            plt.scatter(cone_x, cone_y, color='green')

        # Plot car as point at origin
        plt.scatter(0, 0, color='black')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, self.sight)
        plt.show()

    def plot_absolute(self, cone):
        self._plot_absolute(cone)

    def plot_relative(self, cone):
        self._plot_relative(cone)

if __name__ == "__main__":
    cone1 = Cone(50, -8.0525, 0.9874, 5, 3, 1).positions
    cone2 = Cone(75.5, -4.8315, 1.26, 5, 3, 1).positions
    cone3 = Cone(99, -8.0525, 1.5, 5, 3, 1).positions
    cone_road1 = Cone(62, -3.221, 0, 9, 6, 2).positions
    cone_road211 = Cone(2, -5.542, 0, 10, 6, 2).positions
    cone_rode212 = Cone(56, -4.121, 0, 2, 6, 2).positions
    cone_road221 = Cone(99, -5.542, 0, 10, 6, 2).positions
    cone_road222 = Cone(99, -4.121, 0, 2, 6, 2).positions
    cone_road3 = Cone(75.5, -8.0525, 0.5, 5, 3, 2).positions
    cone_road4 = Cone(2, -9.663, 0, 26, 6, 2).positions
    cone = np.concatenate((cone1, cone2, cone3, cone_road1, cone_road211,cone_rode212, cone_road221, cone_road222, cone_road3, cone_road4))
    vehicle = Car(x=75, y=-8.0525, yaw=0, sight=10)
    print("Max cones in sight: ", vehicle.get_max_cones_in_sight(3))
    a=vehicle.check_collision(cone)
    print(a)
    b=vehicle.get_cone_in_sight(cone)
    print(b)
    vehicle.plot_absolute(cone)
    vehicle.plot_relative(cone)
    for i in range(50):
        vehicle = Car(x=50+i, y=-10.0525, yaw=0, sight=10)
        a=vehicle.check_collision(cone)
        print(50+i, a)
        b=vehicle.get_cone_in_sight(cone)

