import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Point


class Cone:
    def __init__(self, type):
        self.type = type
        if type == 1:
            self.cones = self.create_DLC_cone()
        elif type == 2:
            self.cones = self.create_SLALOM_cone()
        elif type == 3:
            self.cones = self.create_SLALOM_cone2()
        else:
            self.cones = self.create_DLC_cone()
#        self.cones_rel = self.to_relative_coordinates()

    def create_cone(self, sections):
        cones = []
        for section in sections:
            for i in range(section['num']):  # Each section has 5 pairs
                x_base = section['start'] + section['gap'] * i
                y1 = section['y_offset'] - section['cone_dist'] / 2
                y2 = section['y_offset'] + section['cone_dist'] / 2
                cones.extend([[x_base, y1], [x_base, y2]])

        return np.array(cones)

    def create_DLC_cone(self):
        sections = [
            {'start': 0, 'gap': 5, 'cone_dist': 1.9748, 'num': 10, 'y_offset': -8.0525},
            {'start': 50, 'gap': 3, 'cone_dist': 1.9748, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 64.7, 'gap': 2.7, 'cone_dist': 5.4684, 'num': 4, 'y_offset': -6.3057},
            {'start': 75.5, 'gap': 2.75, 'cone_dist': 2.52, 'num': 5, 'y_offset': -4.8315}, #
            {'start': 89, 'gap': 2.5, 'cone_dist': 5.981, 'num': 4, 'y_offset': -6.562},
            {'start': 99, 'gap': 3, 'cone_dist': 3, 'num': 5, 'y_offset': -8.0525}, #
            {'start': 111, 'gap': 5, 'cone_dist': 3, 'num': 20, 'y_offset': -8.0525}
        ]

        return self.create_cone(sections)

    def create_SLALOM_cone(self):
        sections = [
            {'start': 0, 'gap': 5, 'cone_dist': 2, 'num': 18, 'y_offset': -5.25},
            {'start': 85, 'gap': 5, 'cone_dist': 5.25, 'num': 2, 'y_offset': -2.625}, #
            {'start': 100, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -2.625}, #
            {'start': 110, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -5.25},
            {'start': 120, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -5.25},
            {'start': 130, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -7.875}, #
            {'start': 140, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -5.25},
            {'start': 150, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -5.25},
            {'start': 400, 'gap': 10, 'cone_dist': 5.25, 'num': 15, 'y_offset': -5.25}
        ]
        return self.create_cone(sections)
    def create_SLALOM_cone2(self):
        sections = [
            {'start': 100, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -2.625}, #
            {'start': 130, 'gap': 60, 'cone_dist': 5.25, 'num': 5, 'y_offset': -7.875} #
        ]
        return self.create_cone(sections)
    def see_cone(self):
        plt.scatter(self.cones[:,0], self.cones[:, 1])
        plt.axis("equal")
        plt.xlim([50, 100])
        plt.show()

    def cone_in_sight(self, carx, sight):
        return np.array([cone for cone in self.cones if carx - 2.1976004311961135 <= cone[0]][:sight*2])

    def to_relative_coordinates(self, carx, cary, caryaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - carx
            dy = point[1] - cary

            rotated_x = dx * np.cos(-caryaw) - dy * np.sin(-caryaw)
            rotated_y = dx * np.sin(-caryaw) + dy * np.cos(-caryaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)

    def pass_cone_line(self, cones_rel):
        width = 1.568
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135
        dist = 0
        check = 0

        car_upper_line = LineString([(-l1, width/2), (l2, width/2)])
        car_lower_line = LineString([(-l1, -width/2), (l2, -width/2)])

        for i in range(0, len(cones_rel)-1, 2):
            cone1 = cones_rel[i]
            cone2 = cones_rel[i+1]
            line = LineString([cone1, cone2])
            line_center = [(cone1[0] + cone2[0]) / 2, (cone1[1] + cone2[1]) / 2]
           #자동차 영역에 있는지 확인
            if car_upper_line.intersects(line) or car_lower_line.intersects(line):
                check = 1
                # 연결선 중 하나라도 모두 교차하지 않을 경우
                if not (car_upper_line.intersects(line) and car_lower_line.intersects(line)):
                    return 1
        if check == 0:
            pass
        return 0

    def distance_to_closest_cone_line(self, cones_rel):
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135
        cones_rel = [cones_rel[idx] for idx in range(0, len(cones_rel), 2) if -l1 <= cones_rel[idx][0] <= l2]
        min_distance = float('inf')
        min_distance2 = float('inf')
        flag = 0
        flag2 = 0
        width = 1.568

        print(cones_rel)


    def check_collision(self, cones_rel):
        width = 1.568
        length = 4.3
        l1, l2 = 2.1976004311961135, 4.3 - 2.1976004311961135
        for cx, cy in cones_rel:
            if -l1 <= cx <= l2 and -width/2 <= cy <= width/2:
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

        ax.set_xlim(0, 400)
        ax.set_ylim(-10, 0)
        plt.axis('equal')
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

        ax.set_xlim(-20,40)
        ax.set_ylim(-10, 10)
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
    cones = Cone(3)
    print(cones.cones[1, 1])
    #cones.see_cone()
    carx, cary, caryaw = 50, -7.5, 0.2
    cone_sight = cones.cone_in_sight(carx, 5)
    cones_rel = cones.to_relative_coordinates(carx, cary, caryaw, cones.cones)
    print(cones.pass_cone_line(cones_rel))
    print(cones.distance_to_closest_cone_line(cones_rel))
    col = cones.check_collision(cones_rel)
    #print(col)
    #print(cones_rel)
    cones.plot_shapes_abs(carx, cary)
    cones.plot_shapes_rel(carx, cary)
