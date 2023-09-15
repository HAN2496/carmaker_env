import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from shapely.geometry import Polygon, Point


class Cone:
    def __init__(self):
        self.cone_arr = np.array([[i, -5.25] for i in range(100, 400, 30)])

    def cone_in_sight(self, carx, sight):
        return np.array([cone for idx, cone in enumerate(self.cone_arr) if carx - 2.1976004311961135 <= cone[0]][:sight])

    def check_collsion(self, cones_rel):
        width = 1.568
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135
        cone_r = 0.2

        filtered_cones_rel = [cone for cone in cones_rel
                              if (-l1 - cone_r <= cone[0] <= l2 + cone_r) and (-width/2 - cone_r <= cone[1] <= width/2 + cone_r)]

        if not filtered_cones_rel:
            return 0
        print(filtered_cones_rel)

        for conex, coney in filtered_cones_rel:
            if (-l1 <= conex <= l2) and (-width/2 <= coney <= width/2):
                return 1
            car_edge = [[-l1, -width / 2], [l2, -width / 2], [l2, width / 2], [-l1, width / 2]]
            car_line = Polygon(car_edge)
            cone = Point((conex, coney))
            if cone.distance(car_line) <= cone_r:
                return 1

        return 0
    def to_relative_coordinates(self, carx, cary, caryaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - carx
            dy = point[1] - cary

            rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
            rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def plot_shapes_abs(self, carx, cary):
        width, height = 4.3, 1.568
        init_x = 2.1976004311961135
        init_y = height / 2
        r = 0.4 / 2

        fig, ax = plt.subplots()

        rect = patches.Rectangle((carx - init_x, cary - init_y), width, height, angle=np.rad2deg(caryaw), rotation_point=(carx, cary), linewidth=1, edgecolor='r', facecolor='none')
        plt.scatter(carx, cary)
        ax.add_patch(rect)

        # Plot a circle with center at (cx, cy) and radius r
        for cx, cy in self.cone_arr:
            circle = patches.Circle((cx, cy), 0.1, linewidth=1, edgecolor='b', facecolor='none')
            plt.scatter(cx, cy)
            ax.add_patch(circle)


        ax.set_xlim(0, 400)
        ax.set_ylim(-10, 0)
        plt.axis('equal')
        plt.show()

    def plot_shapes_rel(self, carx, cary):
        cones_rel = self.to_relative_coordinates(carx, cary, caryaw, self.cone_arr)
        width, height = 4.3, 1.568
        init_x = 2.1976004311961135
        init_y = height / 2
        r = 0.4 / 2

        fig, ax = plt.subplots()

        rect = patches.Rectangle((0 - init_x, 0 - init_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        plt.scatter(-init_x, -init_y)
        plt.annotate(f'({round(-init_x, 3)}, {round(-init_y, 3)})', (-init_x, -init_y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.scatter(0, 0)
        ax.add_patch(rect)

        # Plot a circle with center at (cx, cy) and radius r
        for cx, cy in cones_rel:
            plt.scatter(cx, cy)
            cx = round(cx, 3)
            cy = round(cy, 3)
            plt.annotate(f'({cx}, {cy})', (cx, cy), textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_xlim(-20,40)
        ax.set_ylim(-10, 10)
        plt.show()

if __name__ == "__main__":
    cone_abs = Cone()
    carx, cary, caryaw = 100, -6.235, 0
    sight = 3
    print(cone_abs.cone_in_sight(carx, sight))
    cone_rel = cone_abs.to_relative_coordinates(carx, cary, caryaw, cone_abs.cone_arr)
    #print(cone_abs.plot_shapes_abs(carx, cary))
    cone_abs.plot_shapes_rel(carx, cary)
    print(cone_abs.check_collsion(cone_rel))
