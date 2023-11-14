import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
import matplotlib.pyplot as plt
from common_functions import *
from carmaker_cone import *

CONER = 0.2
CARWIDTH = 1.8
CARLENGTH = 4
DIST_FROM_AXIS = (CARWIDTH + 1) / 2 + CONER
XSIZE, YSZIE = 10, 10

class Road:
    def __init__(self, road_type):
        self.road_type = road_type

        self.cone = Cone(road_type=road_type)

        if road_type == "DLC":
            self.create_DLC_road()
            self.cone = Cone(road_type=road_type)
        elif road_type == "SLALOM":
            self.create_SLALOM_road()
            self.cone = Cone(road_type=road_type)
        elif road_type == "SLALOM2":
            self.create_SLALOM_road()
            self.cone = Cone(road_type=road_type)

    def create_SLALOM_road(self):
        self.road_length = 500
        if self.road_type == "SLALOM":
            self.road_width = -20

        elif self.road_type == "SLALOM2":
            self.road_width = -50
        else:
            print("Error in carmaker_cone, create_SLALOM_road function")
            sys.exit(1)

        y_mid = self.road_width / 2
        cone_dist = 3
        y_upper = y_mid + cone_dist
        y_lower = y_mid - cone_dist
        vertices1 = [[100 + 30 * i, y_mid - (i % 2 - 1) * cone_dist - 2 * (i % 2 - 0.5) * np.sqrt(2) * CONER] for i in range(10)]
        vertices_upper = np.array(vertices1 + [[385, + y_upper - DIST_FROM_AXIS], [510, + y_upper - DIST_FROM_AXIS],
                                           [510, 15], [0, 15], [0, + y_upper - DIST_FROM_AXIS],
                                           [85, + y_upper - DIST_FROM_AXIS]])
        vertices2 = [[100 + 30 * i, y_mid - (i % 2) * cone_dist - 2 * (i % 2 - 0.5) * np.sqrt(2) * CONER] for i in range(10)]
        vertices_under = np.array(vertices2 + [[385, + y_lower + DIST_FROM_AXIS], [510, + y_lower + DIST_FROM_AXIS], [510, self.road_width - 15],
                                           [0, self.road_width - 15], [0, y_lower + DIST_FROM_AXIS],
                                           [85, y_lower + DIST_FROM_AXIS]])
        self.forbbiden_area1 = Polygon(vertices_upper)
        self.forbbiden_area2 = Polygon(vertices_under)
        self.forbidden_line1 = vertices1
        self.forbidden_line2 = vertices2

        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])

        cone_vertics = np.array(
            [[0, y_upper - DIST_FROM_AXIS], [85, y_upper - DIST_FROM_AXIS]]
            + vertices1
            + [[385, y_upper - DIST_FROM_AXIS], [510, y_upper - DIST_FROM_AXIS], [510, y_lower + DIST_FROM_AXIS], [385, y_lower + DIST_FROM_AXIS]]
            + vertices2[::-1]
            + [[85, y_lower + DIST_FROM_AXIS], [0, y_lower + DIST_FROM_AXIS]]
        )
        self.cone_boundary = Polygon(cone_vertics)

    def create_DLC_road(self):
        self.road_length = 161
        self.road_width = -20
        vertices1 = [
            (0, -8.885 - CONER), (62, -8.885 - CONER), (62, -5.085 - CONER), (99, -5.085 - CONER), (99, -8.885 - CONER),
            (161, -8.85 - CONER), (161, 0), (0, 0), (0, -8.885)
        ]
        vertices2 = [
            (0, -11.115 + CONER), (75.5, -11.115 + CONER), (75.5, -7.885 + CONER), (86.5, -7.885 + CONER),
            (86.5, -11.885 + CONER), (161, -11.885 + CONER), (161, -20), (0, -20), (0, -11.115)
        ]
        self.forbbiden_area1 = Polygon(vertices1)
        self.forbbiden_area2 = Polygon(vertices2)
        self.forbbiden_line1 = LineString(vertices1[:])
        self.forbbiden_line2 = LineString(vertices2[:])
        self.road_boundary = Polygon(
            [(0, 0), (self.road_length, 0), (self.road_length, self.road_width), (0, self.road_width)
        ])
        self.cone_boundary = Polygon(
            [(0, -8.885 - CONER), (62, -8.885- CONER), (62, -5.085- CONER), (99, -5.085- CONER), (99, -8.885- CONER), (161, -8.885- CONER),
             (161, -11.885 + CONER), (86.5, -11.885 + CONER), (86.5, -7.885 + CONER), (75.5, -7.885 + CONER), (75.5, -11.115 + CONER), (0, -11.115 + CONER)
        ])

    def plot_road(self):
        plt.plot(*self.road_boundary.exterior.xy, label='road boundary', color='blue')
        plt.plot(*self.forbbiden_area1.exterior.xy, label='forbidden area', color='red')
        plt.plot(*self.forbbiden_area2.exterior.xy, color='red')
        #for cone in self.cone_boundary:
        #            plt.plot(*cone.exterior.xy)
#        plt.plot(*self.cone_boundary.exterior.xy, label="Cone Boundary", color='orange')
        plt.fill(*self.cone_boundary.exterior.coords.xy, color='orange', alpha=0.5)
        for idx, cone in enumerate(self.cone.cone_arr):
            x, y = cone[0], cone[1]
            if idx == 0:
                plt.scatter(x, y, color='blue', label='cone')
            else:
                plt.scatter(x, y, color='blue')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    road_type = "SLALOM2"
    cone = Cone(road_type=road_type)
    road = Road(road_type=road_type)
    road.plot_road()
