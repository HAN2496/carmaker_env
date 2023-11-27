"""
Cone, Lane, Car 클래스를 정의하는 코드
"""

from common_functions import *
from shapely import affinity

class Cone:
    def __init__(self, road_type):
        self.cone_r = CONER
        self.road_type = road_type
        self.create_cone()

    def create_cone(self):
        if self.road_type == "DLC":
            self.arr = create_DLC_cone_arr()
            self.shape = self.create_cone_shape()
        elif self.road_type == "SLALOM":
            y_middle = -10
            self.arr = create_SLALOM_cone_arr(y_middle)
            self.shape = self.create_cone_shape()
        elif self.road_type == "SLALOM2":
            y_middle = SLALOM2_Y
            self.arr = create_SLALOM_cone_arr(y_middle)
            self.shape = self.create_cone_shape()
        else:
            raise TypeError("Wrong Road type. Put DLC or SLALOM or SLALOM2")
    def create_cone_shape(self):
        shape = []
        for x, y in self.arr:
            cone_square = [(x - self.cone_r, y - self.cone_r),
                           (x - self.cone_r, y + self.cone_r),
                           (x + self.cone_r, y + self.cone_r),
                           (x + self.cone_r, y - self.cone_r)]
            square = Polygon(cone_square)
            shape.append(square)
        return np.array(shape)

    def plot(self, show=True):
        if len(self.arr) != 0:
            for idx, cone in enumerate(self.arr):
                x, y = cone[0], cone[1]
                if idx == 0:
                    plt.scatter(x, y, color='orange', label='cone')
                else:
                    plt.scatter(x, y, color='orange')
            if show:
                plt.legend()
                plt.show()

class Lane:
    def __init__(self, road_type):
        self.road_type = road_type
        self.create_lane()
    def create_lane(self):
        if self.road_type == "DLC":
            upper_arr = np.array([(0, -8.885 - CONER), (62, -8.885 - CONER), (62, -5.085 - CONER),
                               (99, -5.085 - CONER), (99, -8.885 - CONER), (200, -8.85 - CONER)])
            lower_arr = np.array([(0, -11.115 + CONER), (75.5, -11.115 + CONER), (75.5, -7.885 + CONER),
                               (86.5, -7.885 + CONER), (86.5, -11.885 + CONER), (200, -11.885 + CONER)])

        elif self.road_type == "SLALOM2":
            upper_arr = np.array([[0, -25 + DIST_FROM_AXIS], [85, -25 + DIST_FROM_AXIS]] + \
                              [[100 + 30 * i, SLALOM2_Y - (i % 2 - 1) * 3 - 2 * (i % 2 - 0.5) * np.sqrt(2) * CONER] for i in range(10)] + \
                              [[400, -25 + DIST_FROM_AXIS], [510, -25 + DIST_FROM_AXIS]])
            lower_arr = np.array([[x, y - 2 * DIST_FROM_AXIS] for x, y in upper_arr])

        else:
            raise TypeError("Wrong Road type. Put DLC or SLALOM2")

        self.upper_arr = upper_arr
        self.lower_arr = lower_arr
        self.upper_shape = LineString(upper_arr)
        self.lower_shape = LineString(lower_arr)
        self.boundary_arr = np.array(upper_arr.tolist() + lower_arr.tolist()[::-1])
        self.boundary_shape = Polygon(self.boundary_arr)

    def plot(self, show=True):
        x, y = self.upper_shape.xy
        plt.plot(x, y, color='black', linewidth=3, label='Upper Lane')
        x, y = self.lower_shape.xy
        plt.plot(x, y, color='black', linewidth=3, label='Lower Lane')
        x, y = self.boundary_shape.exterior.coords.xy
        plt.fill(x, y, color='blue', label = "Lane Boundary", alpha=0.3)
        if show:
            plt.legend()
            plt.show()

class Road:
    def __init__(self, road_type):
        self.road_type = road_type
        self.cone = Cone(road_type=road_type)
        self.lane = Lane(road_type=road_type)
        self.create_road()

    def create_road(self):
        if self.road_type == "DLC":
            self.create_road_DLC()
        elif self.road_type == "SLALOM2":
            self.create_road_SLALOM2()
        else:
            raise TypeError("Wrong Road type. Put DLC or SLALOM2")
        self.shape = self.create_road_shape(self.length, self.width)
        self.forbidden_area = self.shape.difference(self.lane.boundary_shape)

    def create_road_DLC(self):
        self.length = 200
        self.width = -20

    def create_road_SLALOM2(self):
        self.length = 550
        self.width = SLALOM2_Y * 2

    def create_road_shape(self, x, y):
        return Polygon([(0, 0), (x, 0), (x, y), (0, y)])

    def plot(self, show=True):
        self.cone.plot(show=False)
        self.lane.plot(show=False)
        x, y = self.shape.exterior.xy
        plt.plot(x, y, label='Road', color='black')
        for idx, poly in enumerate(self.forbidden_area.geoms):
            x, y = poly.exterior.coords.xy
            if idx == 0:
                plt.fill(x, y, color='Pink', label = "Forbidden Area", alpha=0.3)
            else:
                plt.fill(x, y, color='Pink', alpha=0.3)
        if show:
            plt.legend()
            plt.show()


class Car:
    def __init__(self):
        self.length = CARLENGTH
        self.width = CARWIDTH
        self.reset_car()

    def reset_car(self):
        self.carx = 2.9855712
        self.cary = -10
        self.caryaw = 0
        self.carv = 13.8888889

    def move_car(self, angle):
        angle = angle[0]
        self.caryaw += angle * 0.01
        self.carx += np.cos(self.caryaw) * self.carv * 0.01
        self.cary += np.sin(self.caryaw) * self.carv * 0.01

    def shape_car(self, carx, cary, caryaw):
        self.carx = carx
        self.cary = cary
        self.caryaw = caryaw
        half_length = self.length / 2.0
        half_width = self.width / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        car_shape = Polygon(corners)
        car_shape = affinity.rotate(car_shape, caryaw, origin='center', use_radians=False)
        car_shape = affinity.translate(car_shape, carx, cary)

        return car_shape


if __name__ == "__main__":
    road_type = "DLC"
    cone = Cone(road_type=road_type)
    lane = Lane(road_type=road_type)
    road = Road(road_type=road_type)
    road.plot()
