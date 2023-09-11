import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Cone:
    def __init__(self, type):
        self.type = type
        if type == 1:
            self.cones = self.create_DLC_cone()
        else:
            self.cones = self.create_DLC_cone()
#        self.cones_rel = self.to_relative_coordinates()

    def create_DLC_cone(self):
        cones = np.array([
            [50 + 3 * 0, -8.0525 - 1.9748 / 2], [50 + 3 * 0, -8.0525 + 1.9748 / 2],
            [50 + 3 * 1, -8.0525 - 1.9748 / 2], [50 + 3 * 1, -8.0525 + 1.9748 / 2],
            [50 + 3 * 2, -8.0525 - 1.9748 / 2], [50 + 3 * 2, -8.0525 + 1.9748 / 2],
            [50 + 3 * 3, -8.0525 - 1.9748 / 2], [50 + 3 * 3, -8.0525 + 1.9748 / 2],
            [50 + 3 * 4, -8.0525 - 1.9748 / 2], [50 + 3 * 4, -8.0525 + 1.9748 / 2],
            [75.5 + 2.75 * 0, -4.8315 - 2.52 / 2], [75.5 + 2.75 * 0, -4.8315 + 2.52 / 2],
            [75.5 + 2.75 * 1, -4.8315 - 2.52 / 2], [75.5 + 2.75 * 1, -4.8315 + 2.52 / 2],
            [75.5 + 2.75 * 2, -4.8315 - 2.52 / 2], [75.5 + 2.75 * 2, -4.8315 + 2.52 / 2],
            [75.5 + 2.75 * 3, -4.8315 - 2.52 / 2], [75.5 + 2.75 * 3, -4.8315 + 2.52 / 2],
            [75.5 + 2.75 * 4, -4.8315 - 2.52 / 2], [75.5 + 2.75 * 4, -4.8315 + 2.52 / 2],
            [99 + 3 * 0, -8.0525 - 3 / 2], [99 + 3 * 0, -8.0525 + 3 / 2],
            [99 + 3 * 1, -8.0525 - 3 / 2], [99 + 3 * 1, -8.0525 + 3 / 2],
            [99 + 3 * 2, -8.0525 - 3 / 2], [99 + 3 * 2, -8.0525 + 3 / 2],
            [99 + 3 * 3, -8.0525 - 3 / 2], [99 + 3 * 3, -8.0525 + 3 / 2],
            [99 + 3 * 4, -8.0525 - 3 / 2], [99 + 3 * 4, -8.0525 + 3 / 2],
            [161 + 3 * 0, -8.0525 - 3 / 2], [161 + 3 * 0, -8.0525 + 3 / 2],
            [161 + 3 * 1, -8.0525 - 3 / 2], [161 + 3 * 1, -8.0525 + 3 / 2],
            [161 + 3 * 2, -8.0525 - 3 / 2], [161 + 3 * 2, -8.0525 + 3 / 2],
            [161 + 3 * 3, -8.0525 - 3 / 2], [161 + 3 * 3, -8.0525 + 3 / 2],
            [161 + 3 * 4, -8.0525 - 3 / 2], [161 + 3 * 4, -8.0525 + 3 / 2]
        ])
        return cones

    def cone_in_sight(self, carx, sight):
        for idx, (cx, cy) in enumerate(self.cones):
            if carx <= cx:
                return self.cones[idx:idx+sight*2+1, :]


    def pass_cone_line(self, carx, cary, caryaw):
        cones_rel = self.to_relative_coordinates(carx, cary, caryaw, self.cones)
        width = 1.568
        length = 4.3
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135

        # Find pairs of cones that have the same x-coordinate as the car
        cone_pairs = [(cones_rel[i], cones_rel[i+1]) for i in range(0, len(cones_rel)-1, 2) if abs(cones_rel[i][0]) < length/2]
        print(cone_pairs)

        for cone1, cone2 in cone_pairs:
            # Check if the car is between the cones in y-coordinate
            if (cone1[1] < -width/2 and cone2[1] > width/2) or (cone1[1] > width/2 and cone2[1] < -width/2):
                # Check if the car is not passing directly between the cones
                if not (-l1 <= cone1[0] <= l2 and -width/2 <= cone1[1] <= width/2) and not (-l1 <= cone2[0] <= l2 and -width/2 <= cone2[1] <= width/2):
                    return 1

        # If the car isn't passing between any pair of cones, or is passing directly between the cones
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

    def check_collision(self, cones_rel):
        width = 1.568
        length = 4.3
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135
        for cx, cy in cones_rel:
            if -l1 <= cx <= l2 and -width/2 <= cy <= width/2:
                print(cx, cy)
                return 1
        return 0


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
        for cx, cy in self.cones:
            circle = patches.Circle((cx, cy), 0.1, linewidth=1, edgecolor='b', facecolor='none')
            plt.scatter(cx, cy)
            ax.add_patch(circle)

        cone_num = self.cones.shape[0]
        half_cone_num = int(cone_num / 2)
        for i in range(half_cone_num):
            tmp1 = i * 2
            x_points = self.cones[tmp1:tmp1+2][:, 0]
            y_points = self.cones[tmp1:tmp1+2][:, 1]
            plt.plot(x_points, y_points, '-')

        ax.set_xlim(carx-10, carx+10)
        ax.set_ylim(-10, 0)
        plt.show()

    def plot_shapes_rel(self, carx, cary):
        cones_rel = self.to_relative_coordinates(carx, cary, caryaw, self.cones)
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


        cone_num = self.cones.shape[0]
        half_cone_num = int(cone_num / 2)
        for i in range(half_cone_num):
            tmp1 = i * 2
            x_points = cones_rel[tmp1:tmp1+2][:, 0]
            y_points = cones_rel[tmp1:tmp1+2][:, 1]
            plt.plot(x_points, y_points, '-')

        ax.set_xlim(-5,5)
        ax.set_ylim(-5, 5)
        plt.show()
"""
    def check_collision(self, carx, cary, caryaw):
        cones_rel = self.to_relative_coordinates(carx, cary, caryaw, self.cones)
        width = 1.568
        length = 4.3
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135
        r = 0.4 / 2
        ps = [[-l1, - width/2], [-l1, width/2], [l2, width/2], [l2, -width/2]]
        for cx, cy in cones_rel:
            if -l1 <= cx <= l2 and -width/2 <= width/2:
                return 1
            if - l1 - r <= cx <= l2 + r and -width / 2 - r <= cy <= width / 2 + r:
                for px, py in ps:
                    if math.sqrt((px - cx) ** 2 + (py - cy) **2) <= r:
                        return 1

        return 0
"""

if __name__ == "__main__":
    cones = Cone(1)
    carx, cary, caryaw = 60, -8, 0.5
    cone_sight = cones.cone_in_sight(carx, 5)
    print(cone_sight)
    print(cones.pass_cone_line(carx, cary, caryaw))
    cones_rel = cones.to_relative_coordinates(carx, cary, caryaw, cones.cones)
    col = cones.check_collision(cones_rel)
    #print(col)
    #print(cones_rel)
    #cones.plot_shapes_abs(carx, cary)
    cones.plot_shapes_rel(carx, cary)
