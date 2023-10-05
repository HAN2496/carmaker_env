import numpy as np

    def shape_car(self, caryaw):
        half_length = 4.3 / 2.0
        half_width = 1.568 / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        car_shape = Polygon(corners)
        car_shape = affinity.rotate(car_shape, caryaw, origin='center', use_radians=False)

        return car_shape
    def check_collision(self, caryaw, cones_rel):
        car = self.shape_car(caryaw)
        np.array([Point(cx, cy).buffer(0.2) for cx, cy in cones_rel])
        for conex, coney in cones_rel:
            cone = Point(conex, coney).buffer(0.2)
            if car.intersects(cone):
                print('collsiion')
                return 1
        return 0

    def to_relative_coordinates(self, x, y, yaw, arr):
        relative_coords = []

        for point in arr:
            dx = point[0] - x
            dy = point[1] - y

            rotated_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
            rotated_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)

            relative_coords.append((rotated_x, rotated_y))

        return np.array(relative_coords)
def create_cone(sections):
    conex = []
    coney = []
    for section in sections:
        for i in range(section['num']):  # Each section has 5 pairs
            x_base = section['start'] + section['gap'] * i
            y = section['y_offset']
            conex.extend([x_base])
            coney.extend([y])

    data = np.array([conex, coney]).T
    data_sorted = data[data[:, 0].argsort()]

    xcoords_sorted = data_sorted[:, 0]
    ycoords_sorted = data_sorted[:, 1]

    return np.column_stack((xcoords_sorted, ycoords_sorted))


def create_SLALOM_cone():
    sections = [
        {'start': 100, 'gap': 60, 'num': 5, 'y_offset': -5.25},  #
        {'start': 130, 'gap': 60, 'num': 5, 'y_offset': -5.25},  #
        {'start': 600, 'gap': 10, 'num': 5, 'y_offset': -5.25}
    ]
    return create_cone(sections)
def find_cone(cones, carx, sight):
    cone_in_sight = []
    for conex, coney in cones:
        if carx - 2.1976004311961135 <= conex <= carx + sight:
            cone_in_sight.append([conex, coney])

    return cone_in_sight
def cone_in_sight(cones, carx, sight):
    return np.array([cone for cone in cones if carx - 2.1976004311961135 <= cone[0]][:sight])

a = create_SLALOM_cone()
carx = 200
sight = 5
print(a)
print(find_cone(a, carx, sight))
print(cone_in_sight(a, carx, sight))
