import numpy as np
from shapely.geometry import Polygon, Point, LinearRing, LineString, MultiLineString, GeometryCollection
from shapely.affinity import rotate
import matplotlib.pyplot as plt

class Forbidden:
    def __init__(self):
        vertices1 = [
            (0, -6.442),
            (62, -6.442),
            (62, -3.221),
            (99, -3.221),
            (99, -6.442),
            (161, -6.442),
            (161, 0),
            (0, 0),
            (0, -6.442)
        ]
        vertices2 = [
            (0, -9.663),
            (75.5, -9.663),
            (75.5, -6.442),
            (86.5, -6.442),
            (86.5, -9.663),
            (161, -9.663),
            (161, -12.884),
            (0, -12.884),
            (0, -9.663)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])


class Cone:
    def __init__(self, x, y, offset, num_cones, gap):
        self.x = x
        self.y = y
        self.offset = offset
        self.num_cones = num_cones
        self.gap = gap
        positions = []

        if offset == 0:
            for i in range(num_cones):
                positions.append((x, y))
                x += gap
        else:
            for i in range(num_cones):
                positions.append((x, y - offset))
                positions.append((x, y + offset))
                x += gap
        self.positions = np.array(positions)

class Car:
    def __init__(self, x, y, yaw, sight=10, show=False):
        # initializing car position.
        self.car_width = 1.568
        self.car_length = 4.3
        self.x = x
        self.y = y
        self.yaw = yaw
        self.show = show
        self.sight = sight
        self.data_len = self.sight * 3 * 5

        self.vehicle_polygon = Polygon([
            (self.x, self.y - self.car_width / 2),
            (self.x, self.y + self.car_width / 2),
            (self.x - self.car_length, self.y + self.car_width / 2),
            (self.x - self.car_length, self.y -self.car_width / 2)
        ])
        self.rotated_vehicle_polygon = rotate(self.vehicle_polygon, self.yaw, origin=(self.x, self.y), use_radians=True)

    def check_collision(self, cone, forbbiden):
        col = 0

        for cone_x, cone_y in cone:
            cone_polygon = Point(cone_x, cone_y)

            if self.rotated_vehicle_polygon.intersects(cone_polygon):
                col = 1

        if self.rotated_vehicle_polygon.intersects(
                forbbiden.forbbiden_area1) or self.rotated_vehicle_polygon.intersects(
                forbbiden.forbbiden_area2):
            col = 2

        if self.show == True:
            plt.scatter(self.x, self.y, color='black')
            plt.plot(*self.rotated_vehicle_polygon.exterior.xy, color='black', label='Vehicle')
            plt.plot(*forbbiden.forbbiden_area1.exterior.xy, color='red', label='Forbidden Area')
            plt.plot(*forbbiden.forbbiden_area2.exterior.xy, color='red', label='Forbidden Area')

            for cone_x, cone_y in cone:
                plt.scatter(cone_x, cone_y, color='green', label='Cone')

            plt.gca().set_aspect('equal', adjustable='box')
            plt.ylim(-14, 1)
            plt.show()

#        print(col)

        return col




    def see_object(self, cone, forbidden, show=False, car_view=True):

        def rotate_coords(arr):
            rad = -self.yaw
            R = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
            arr -= np.array([self.x, self.y])
            return np.dot(arr, R)


        cone_arr = []
        for cone_x, cone_y in cone:
            rel_cone = rotate_coords(np.array([cone_x, cone_y]))
            if 0 <= rel_cone[0] <= self.sight:
                cone_arr.append([cone_x, cone_y])
        if np.array(cone_arr).size > 0:
            cone_arr_rel = rotate_coords(np.array(cone_arr))
        else:
            cone_arr_rel = np.empty((0,2))
            cone_arr = np.empty((0, 2))


        total_view_arr = np.array(cone_arr)
        total_view_arr_rel = cone_arr_rel

        rotate_sight_line_arr = []
        sight_line_arr = []
        forbidden_line_arr_rel = []
        i=0
        last_data1 = []
        last_data2 = []

        while np.size(total_view_arr_rel) <= self.data_len:
            empty=0
            forbidden_line_arr = []
            sight_line = LineString([
                (self.x + i, 160),
                (self.x + i, -160)
            ])

            rotate_sight_line = rotate(sight_line, self.yaw, use_radians=True, origin=(self.x, self.y))
            rotate_sight_line_arr.append(rotate_sight_line)

            if self.show:
                sight_line_arr.append(sight_line)

            if forbidden.forbbiden_line1.intersects(rotate_sight_line):
                intersect_point1 = forbidden.forbbiden_line1.intersection(rotate_sight_line)
            else:
                intersect_point1 = GeometryCollection()

            if forbidden.forbbiden_line2.intersects(rotate_sight_line):
                intersect_point2 = forbidden.forbbiden_line2.intersection(rotate_sight_line)
            else:
                intersect_point2 = GeometryCollection()


            if intersect_point1.geom_type == 'MultiPoint':
                for point in intersect_point1.geoms:
                    forbidden_line_arr.append([point.x, point.y])
                    last_data1 = forbidden_line_arr[-1]
            elif intersect_point1.is_empty:
                empty += 1
            elif intersect_point1.geom_type == 'LineString':
                forbidden_line_arr.extend(list(intersect_point1.coords))
            else:
                forbidden_line_arr.append([intersect_point1.x, intersect_point1.y])
                last_data1 = forbidden_line_arr[-1]

            if intersect_point2.geom_type == 'MultiPoint':
                for point in intersect_point2.geoms:
                    forbidden_line_arr.append([point.x, point.y])
                last_data2 = forbidden_line_arr[-1]
            elif intersect_point2.is_empty:
                empty += 1
            elif intersect_point2.geom_type == 'LineString':
                forbidden_line_arr.extend(list(intersect_point2.coords))
            else:
                forbidden_line_arr.append([intersect_point2.x, intersect_point2.y])
                last_data2 = forbidden_line_arr[-1]


            if empty == 2:
                if last_data1:
                    forbidden_line_arr.append(last_data1)
                if last_data2:
                    forbidden_line_arr.append(last_data2)
            else:
                i+=1
            forbidden_line_arr_rel = rotate_coords(np.array(forbidden_line_arr))
            sort_indices = np.argsort(forbidden_line_arr_rel[:, 0])
            forbidden_line_arr_rel = forbidden_line_arr_rel[sort_indices]
            total_view_arr_rel = np.concatenate((total_view_arr_rel, forbidden_line_arr_rel), axis=0)
            total_view_arr = np.concatenate((total_view_arr, forbidden_line_arr), axis=0)


        if self.data_len < np.size(total_view_arr_rel):
            total_view_arr_rel = total_view_arr_rel[:self.data_len//2]
#            print(self.data_len, np.size(total_view_arr_rel), total_view_arr_rel.shape)
        if self.data_len < np.size(total_view_arr):
            total_view_arr = total_view_arr[:self.data_len//2]



        if show and car_view:
            plt.scatter(0, 0, color='black')

            for sight_line in sight_line_arr:
                plt.plot(np.array(sight_line.coords.xy[0]) - self.x, np.array(sight_line.xy[1]) - self.y, linestyle='--', color='blue')
            for line_x, line_y in total_view_arr_rel:
                plt.scatter(line_x, line_y, color='red')

            for cone_x, cone_y in cone_arr_rel:
                plt.scatter(cone_x, cone_y, color='green')

            plt.gca().set_aspect('equal', adjustable='box')
            plt.ylim(-20, 20)
            plt.show()


        elif show and car_view==False:
            plt.plot(np.array(self.rotated_vehicle_polygon.exterior.xy[0]), np.array(self.rotated_vehicle_polygon.exterior.xy[1]), color='black', label='Vehicle')
            plt.scatter(self.x, self.y, color='black')

            for rot_sight_line in rotate_sight_line_arr:
                plt.plot(np.array(rot_sight_line.coords.xy[0]), np.array(rot_sight_line.xy[1]), color='purple')

            for line_x, line_y in total_view_arr:
                plt.scatter(line_x, line_y, color='red')

            for cone_x, cone_y in np.array(cone_arr):
                plt.scatter(cone_x, cone_y, color='green')

            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

#        print(cone_arr_rel)

#        print(total_view_arr_rel)
#        print(np.size(total_view_arr_rel))

        return total_view_arr_rel.flatten()


if __name__ == "__main__":
    cone1 = Cone(50, -8.0525, 0.9874, 5, 3).positions
    cone2 = Cone(75.5, -4.8315, 1.26, 5, 3).positions
    cone3 = Cone(99, -8.0525, 1.5, 5, 3).positions
    cone = np.concatenate((cone1, cone2, cone3))
    area = Forbidden()
    vehicle = Car(x=80.6, y=-10.1, yaw=0.3, sight=10, show=True)
    vehicle.check_collision(cone, area)
    k = vehicle.see_object(cone, area, show=True, car_view=True)
