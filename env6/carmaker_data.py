from shapely import affinity
from carmaker_cone import *
import pandas as pd
from scipy.spatial import KDTree
import pygame
from carmaker_trajectory3 import Trajectory
from carmaker_trajectory_low import Trajectory as LowTrajectory
import time

class Data:
    def __init__(self, road_type, env_num=2, show=False):
        self.road_type = road_type
        self.env_num = env_num
        self.car = Car()
        self.road = Road(road_type=road_type)
        self.traj = Trajectory(road_type=road_type)
        self.test_num = 0

        self.do_render = False
        if env_num == 0 and show:
            self.do_render = True

        if self.road_type == "DLC":
            self.XSIZE, self.YSIZE = 9, 9
        elif self.road_type == "SLALOM" or self.road_type == "SLALOM2":
            self.XSIZE, self.YSIZE = 2, 5


        self._init()

    def _init(self):
        x, y = init_car_pos(road_type=self.road_type)
        arr = [
            0, 0, x, y, 0, 13.8889,
            0, 0, 0,
            0, 0,
            0, 0, 0, 0, 0, 0
        ]
        self.put_simul_data(arr)

        if self.do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.road.length * self.XSIZE, - self.road.width * self.YSIZE))
            pygame.display.set_caption("B level Environment")
            self.render()

        self.traj._init_traj()
        self.devDist, self.devAng = 0, 0
        self.get_lookahead_traj_abs()
        self.reward = 0

    def put_simul_data(self, arr):
        self.simul_data = arr
        self.test_num += 1
        self.time = arr[1]
        self.carx, self.cary, self.caryaw, self.carv = arr[2:6]
        self.steerAng, self.steerVel, self.steerAcc = arr[6:9]
        self.alHori, self.roll = arr[9:11]
        self.rl, self.rr, self.fl, self.fr, self.rl_ext, self.rr_ext = arr[11:]

        self.steer = arr[6:9]
        self.wheel_steer = arr[11:15]
        self.wheel_steer_ext = arr[15:]

        self.devDist, self.devAng = self.traj.calculate_dev(self.carx, self.cary, self.caryaw)
        self.get_lookahead_traj_abs()
        self.car_shape = self.car.shape_car(self.carx, self.cary, self.caryaw)

        if self.test_num % 150 == 0 and self.env_num == 0:
            print(f"Time: {self.time}, Pos : [x: {round(self.carx, 2)}] [y: {round(self.cary, 2)}] Reward : [dist : {round(self.devDist,2)}] [angle : {round(self.devAng, 2)}]")

    def manage_state_low(self):
        lookahead_traj_rel = self.get_lookahead_traj_rel()
        closet_cones = self.get_cones_rel(0).flatten()
        car_data = np.array([self.devDist, self.devAng, self.caryaw, self.carv, self.steerAng, self.steerVel,
                             self.rl, self.rr, self.fl, self.fr, self.rl_ext, self.rr_ext])
        return np.concatenate((car_data, lookahead_traj_rel, closet_cones))

    def manage_reward_low(self):
        dist_reward = abs(self.devDist) * 100
        ang_reward = abs(self.devAng) * 500
        if self.road_type == "DLC" or self.road_type == "SLALOM2":
            col_reward = self.is_car_in_lane() * 1000
        elif self.road_type == "Eight_20m":
            col_reward = 0
        else:
            col_reward = abs(self.alHori) * 1000

        e = - col_reward - dist_reward - ang_reward
        self.reward = e
        return e

    def manage_b(self):
        state = self.manage_state_b()
        info_key = np.array(["num", "time", "x", "y", "yaw", "carv", "ang", "vel", "acc", "alHori", "roll",
                             "rl", "rr", "fl", "fr", "rl_ext", "rr_ext"])
        info = {key: value for key, value in zip(info_key, self.simul_data)}
        done =self.manage_done_b()
        reward = self.manage_reward_b()
        return state, reward, done, info

    def manage_state_b(self):
        lookahead_traj_rel = self.get_lookahead_traj_rel()
        car_data = np.array([self.devDist, self.devAng, self.caryaw, self.carv, self.steerAng, self.steerVel,
                             self.rl, self.rr, self.fl, self.fr])
        if self.road_type == "DLC":
            cones_rel = self.get_cones_rel(pos=[0, 4])
            return np.concatenate((car_data, cones_rel, lookahead_traj_rel))
        else:
            return np.concatenate((car_data, lookahead_traj_rel))

    def manage_reward_b(self):
        last_point = self.lookahead_traj_abs[-1]
        last_shape = Point(last_point[0], last_point[1])
        if not self.road.lane.boundary_shape.intersects(last_shape):
            forbidden_reward = -10000
        else:
            forbidden_reward = 0
        if not self.road.lane.boundary_shape.intersects(last_shape):
            cones_reward = +100
        else:
            cones_reward = 0
        if self.is_car_in_lane():
            car_reward = 0
        else:
            car_reward = -10000

        e = forbidden_reward + cones_reward + car_reward
        self.reward=e
        return e

    def manage_done_b(self):
        x, y = self.lookahead_traj_abs[-1]
        if self.road.shape.intersects(Point(x, y)):
            return False
        else:
            return True

    def get_lookahead_traj_abs(self):
        lookahead_sight = [2 * i for i in range(5)]
        self.lookahead_traj_abs = self.traj.find_traj_points(self.carx, lookahead_sight)

    def get_lookahead_traj_rel(self):
        lookahead_traj_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], self.lookahead_traj_abs).flatten()
        return lookahead_traj_rel

    def get_cones_rel(self, pos):
        if pos == 0:
            ahead_cones = self.road.cone.arr[self.road.cone.arr[:, 0] > self.carx][:1]
            behind_cones = self.road.cone.arr[self.road.cone.arr[:, 0] <= self.carx][:1]
            closest_cones = np.vstack((behind_cones, ahead_cones))
            cones_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], closest_cones).flatten()
            return cones_rel
        """
        pos: [a, b]
        1) a ~ b 사이에 있는 콘의 위치.
        2) if a or b == 0: 현재 차 바로 앞 콘의 위치
        3) if a or b == 3: 현재 차 기준 세 개 앞에 있는 콘의 위치
        """
        if np.shape(pos) != (2, ):
            raise TypeError("Wrong shape of the function get_cones_rel")
        elif abs(pos[1] - pos[0]) == 0:
            raise TypeError("Enter two integer values that differ by at least 1")
        pos = sorted(pos)

        cones = self.road.cone.arr[:, 0]
        car_position_index = np.searchsorted(cones, self.carx)

        start_index = car_position_index + pos[0] - (1 if pos[0] > 0 else 0)
        end_index = car_position_index + pos[1] - (1 if pos[1] > 0 else 0)

        cones_abs = self.road.cone.arr[start_index:end_index]

        cones_rel = to_relative_coordinates([self.carx, self.cary, self.caryaw], cones_abs).flatten()
        return cones_rel

    def is_car_in_lane(self):
        if self.road.lane.boundary_shape.contains(self.car_shape):
            return 0
        return 1

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill(GRAY)

        #콘 렌더링
        for idx, cone in enumerate(self.road.cone.shape):
            x, y = cone.centroid.coords[0]
            pygame.draw.circle(self.screen, ORANGE, (int(x * self.XSIZE), int(-y * self.YSIZE)), 5)

        #Trajectory Points 렌더링
        traj_abs = self.lookahead_traj_abs
        for idx, (trajx, trajy) in enumerate(traj_abs):
            if idx % 1 == 0:
                pygame.draw.circle(self.screen, GREEN, (trajx * self.XSIZE, - trajy * self.YSIZE), 5)

        #텍스트 렌더링. 여기서는 마지막 포인트 렌더링 했음 ㅇㅇ.
        font = pygame.font.SysFont("arial", 15, True, True)

        x = self.carv
        text_str = f"Velocity : ({round(x, 1)})"
        text_surface = font.render(text_str, True, WHITE)

        text_x = self.road.length * self.XSIZE - text_surface.get_width() - self.XSIZE
        text_y = - self.road.width * self.YSIZE - text_surface.get_height() - self.YSIZE

        self.screen.blit(text_surface, (text_x, text_y))

        #차량 렌더링
        half_length = self.car.length * self.XSIZE / 2.0
        half_width = self.car.width * self.YSIZE / 2.0

        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

        rotated_corners = []
        for x, y in corners:
            x_rot = x * np.cos(-self.caryaw) - y * np.sin(-self.caryaw) + self.carx * self.XSIZE
            y_rot = x * np.sin(-self.caryaw) + y * np.cos(-self.caryaw) - self.cary * self.YSIZE
            rotated_corners.append((x_rot, y_rot))

        pygame.draw.polygon(self.screen, RED, rotated_corners)

        pygame.display.flip()


    def plot(self, show=True):
        self.road.plot(show=False)
        self.traj.plot(show=False)
        x, y = self.car.shape_car(self.carx, self.cary, self.caryaw).exterior.xy
        plt.fill(x, y, label="Car", color="gray", alpha=1)
        plt.plot(x, y, color="gray")

        if show:
            plt.legend()
#            plt.axis('equal')
            plt.show()

    def test(self, carx, cary):
        arr = [
            0, 0, carx, cary, 0, 13.8889,
            0, 0, 0,
            0, 0,
            0, 0, 0, 0, 0, 0
        ]
        self.put_simul_data(arr)
        action = [1, -1]
        if self.do_render:
            self.render()

if __name__ == "__main__":
    road_type, show = "DLC", True
    data = Data(road_type=road_type, env_num=0, show=show)
    #data.test(100, -10)
    print(data.get_cones_rel([-3, 2]))
    data.plot()

    running = True
    start_time = time.time()
    duration = 10  # 렌더링을 실행할 시간(초)


    pos = 2
    while running:
        data.test(pos, -10)
        # 현재 시간과 시작 시간의 차이가 duration보다 크면 루프를 종료합니다.
        if time.time() - start_time > duration:
            running = False

        # 테스트 데이터 업데이트
        data.test(pos, -10)

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 렌더링 실행
        data.render()
        print(data.is_car_in_lane())
        # 프레임 갱신을 위한 짧은 지연
        time.sleep(0.1)
        pos += 1

    pygame.quit()



    data._init()

    running = True
    start_time = time.time()
    duration = 10  # 렌더링을 실행할 시간(초)

    pos = 2
    while running:
        # 현재 시간과 시작 시간의 차이가 duration보다 크면 루프를 종료합니다.
        if time.time() - start_time > duration:
            running = False

        # 테스트 데이터 업데이트
        data.test(pos, -10)

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 렌더링 실행
        data.render()
        print(data.is_car_in_lane())
        # 프레임 갱신을 위한 짧은 지연
        time.sleep(0.1)
        pos += 1

    pygame.quit()

