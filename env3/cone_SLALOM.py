import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt

cone_r = 0.2
car_width, car_length = 1.568, 4
dist_from_axis = (car_width + 1) / 2 + cone_r



def plot(road, car):
    #plt.figure(figsize=(10, 5))

    # Plot forbidden areas
    plt.plot(*road.cones_boundary.exterior.xy, label="Cone Boundary", color='red')
#    plt.plot(*road.forbbiden_area1.exterior.xy, label="Forbidden Area 1", color='red')
#    plt.plot(*road.forbbiden_area2.exterior.xy, label="Forbidden Area 2", color='blue')
    plt.plot(*road.road_boundary.exterior.xy, label="ROAD BOUNDARY", color='green')

    # Plot cones
    cones_x = road.cones_arr[:, 0]
    cones_y = road.cones_arr[:, 1]
    plt.scatter(cones_x, cones_y, s=10, color='orange', label="Cones")

    # Plot the car
    car_shape = car.shape_car(car.carx, car.cary, car.caryaw)
    plt.plot(*car_shape.exterior.xy, color='blue', label="Car")
    plt.scatter(car.carx, car.cary, color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.gca().invert_yaxis()
    plt.title('Car, Forbidden Areas and Cones')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

class Road:
    def __init__(self):
        self.road_length = 500
        self.road_width = -20
        self.cone_r = 0.2
        self.car_width, car_length = 1.568, 4
        self._forbidden_area()
        self.cones_arr = self.create_cone_arr()
        self.cones_shape = self.create_cone_shape()

    def _forbidden_area(self):
        vertices1 = [
            (0, -8.885), (62, -8.885), (62, -5.085), (99, -5.085), (99, -8.885),
            (161, -8.85), (161, 0), (0, 0), (0, -8.885)
        ]
        vertices2 = [
            (0, -11.885), (75.5, -11.885), (75.5, -7.885), (86.5, -7.885),
            (86.5, -11.885), (161, -11.885), (161, -20), (0, -20), (0, -11.885)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])
        self.cones_boundary = Polygon(
            [(0, -8.885), (62, -8.885), (62, -5.085), (99, -5.085), (99, -8.885), (161, -8.885),
             (161, -11.885), (86.5, -11.885), (86.5, -7.885), (75.5, -7.885), (75.5, -11.115), (0, -11.115)
        ])

    def create_cone_shape(self):
        cones = []
        for i, j in self.cones_arr:
            cone = Point(i, j).buffer(0.2)
            cones.extend(cone)

        return np.array(cones)

    def create_cone_arr(self):
        """
        dist_from_axis = (self.car_width + 1) / 2 + self.cone_r
        def cubic_interpolation(x0, y0, x1, y1, interval=0.01):
            a = 2 * (y0 - y1) / (x1 - x0) ** 3
            b = -3 / 2 * a * (x1 - x0)
            d = y0

            def interpolator(x):
                return a * (x - x0) ** 3 + b * (x - x0) ** 2 + d

            f = interpolator
            x = np.linspace(x0, x1, int((x1 - x0) / interval)+1)
            y = [f(xi) for xi in x]

            return np.array(list(zip(x, y)))

        dist_from_axis_at_straight = 1.1 * car_width + 0.25
        dist_from_axis_at_straight = dist_from_axis_at_straight / 2

        cone_straight_line1_upper = np.array([[x, -5.25 + dist_from_axis_at_straight] for x in np.arange(0, 85, 5)])
        cone_straight_line1_lower = np.array([[x, -5.25 - dist_from_axis_at_straight] for x in np.arange(0, 85, 5)])
        cone_straight_line2_upper = np.array([[x, -5.25 + dist_from_axis_at_straight] for x in np.arange(385, 500, 5)])
        cone_straight_line2_lower = np.array([[x, -5.25 - dist_from_axis_at_straight] for x in np.arange(385, 500, 5)])

        traj_cone_upper = np.vstack((cone_straight_line1_upper, cone_straight_line2_upper))
        traj_cone_lower = np.vstack((cone_straight_line1_lower, cone_straight_line2_lower))
        total_cone = np.vstack((cone_straight_line1_upper, cone_straight_line2_upper, cone_straight_line1_lower, cone_straight_line2_lower))

        cone_interpolate_first_upper = cubic_interpolation(100 - 15, -5.25 + dist_from_axis_at_straight, 100, -5.25 + dist_from_axis + cone_r, interval=2)
        cone_interpolate_last_upper = cubic_interpolation(370, -5.25 - cone_r, 370 + 15, -5.25 + dist_from_axis_at_straight, interval=2)
        cone_interpolate_first_lower = cubic_interpolation(100 - 15, -5.25 - dist_from_axis_at_straight, 100, -5.25 + cone_r, interval=2)
        cone_interpolate_last_lower = cubic_interpolation(370, -5.25 - dist_from_axis - cone_r, 370 + 15, -5.25 - dist_from_axis_at_straight, interval=2)

        traj_cone_upper = np.vstack((traj_cone_upper, cone_interpolate_first_upper, cone_interpolate_last_upper))
        traj_cone_lower = np.vstack((traj_cone_lower, cone_interpolate_first_lower, cone_interpolate_last_lower))
        total_cone = np.vstack((total_cone, cone_interpolate_first_upper, cone_interpolate_last_upper, cone_interpolate_first_lower, cone_interpolate_last_lower))

        for i in range(9):
            dist_from_axis = (car_width + 1) / 2 + cone_r
            if i % 2 == 0:
                interpolated_upper = cubic_interpolation(100 + 30 * i, -5.25 + (dist_from_axis + cone_r), 100 + 30 * (i + 1), -5.25 - cone_r, interval=2)
                interpolated_lower = cubic_interpolation(100 + 30 * i, -5.25 + cone_r, 100 + 30 * (i + 1), -5.25 - (dist_from_axis + cone_r), interval=2)
            else:
                interpolated_upper = cubic_interpolation(100 + 30 * i, -5.25 - cone_r, 100 + 30 * (i + 1), -5.25 + (dist_from_axis + cone_r), interval=2)
                interpolated_lower = cubic_interpolation(100 + 30 * i, -5.25 - (dist_from_axis + cone_r), 100 + 30 * (i + 1), -5.25 + cone_r, interval=2)

            total_cone = np.concatenate((total_cone, interpolated_upper, interpolated_lower), axis=0)
            traj_cone_upper = np.concatenate((traj_cone_upper, interpolated_upper), axis=0)
            traj_cone_lower = np.concatenate((traj_cone_lower, interpolated_lower), axis=0)

        cone = np.array([total_cone[:, 0], total_cone[:, 1]]).T
        cone_sorted = total_cone[total_cone[:, 0].argsort()]

        traj_cone_upper = np.array([traj_cone_upper[:, 0], traj_cone_upper[:, 1]]).T
        self.traj_cone_upper = traj_cone_upper[traj_cone_upper[:, 0].argsort()]

        traj_cone_lower = np.array([traj_cone_lower[:, 0], traj_cone_lower[:, 1]]).T
        self.traj_cone_lower = traj_cone_lower[traj_cone_lower[:, 0].argsort()]
        """
        cones = np.array([100 + 30 * i, -5.25] for i in range(15))

        return np.array(cones)


    def is_car_in_forbidden_area(self, car_shape):

        if car_shape.intersects(self.forbbiden_area1) or car_shape.intersects(self.forbbiden_area2):
            return 1
        else:
            return 0
    def is_car_colliding_with_cones(self, car_shape):
        for cone in self.cones_shape:
            if car_shape.intersects(cone):
                return 1
        return 0

    def is_car_in_road(self, car_shape):
        if not car_shape.intersects(self.road_boundary):
            return 1
        if not self.road_boundary.contains(car_shape):
            return 1
        return 0

    def is_car_in_cone_area(self, car_shape):
        if not car_shape.intersects(self.cones_boundary):
            return 1
        if not self.cones_boundary.contains(car_shape):
            return 1
        return 0

class Car:
    def __init__(self, carx=3, cary=-5.25, caryaw=0, carv=13.8889):
        self.length = 4.3
        self.width = 1.568
        self.carx = carx
        self.cary = cary
        self.caryaw = caryaw
        self.carv = carv

    def reset_car(self):
        self.carx = 3
        self.cary = -10
        self.caryaw = 0

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
