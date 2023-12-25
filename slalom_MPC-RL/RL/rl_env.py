import numpy as np
# import gym
import gymnasium as gym
from casadi import interpolant
from scipy.interpolate import CubicSpline

from environment import SlalomTest
from environment.config import Config
from environment.vehicle_model import VehicleModel
from environment.track import Track
from mpc import MPC

class RLEnv(gym.Env):
    """
    RL environment for training with 2D environment.
    !! Deprecated !!
    """

    def __init__(self, is_render, render_mode=None):
        super(RLEnv, self).__init__()
        # self.is_render = is_render

        self._config = Config()
        self._vehicle_model = VehicleModel(self._config)
        self._track = Track()
        self.controller = MPC(self._vehicle_model, self._track, self._config)
        self.slalom_env = SlalomTest(is_render, render_mode='human')
        
        self.h = self._config.rl.h

        """
        Actions : (6,)
        |     ddt     |  10 points of reference ec  |
        ---------------------------------------------
        | shape: (1,) |         shape: (5,)        |
        """
        self.action_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (6,), # only ddt
            dtype = np.float32
        )

        """
        Observation : Dictinary
        |  vehicle state: Y, psi, vx, vy, omega, delta, dt  |       barrier for length h       |
        ------------------------------------------------------------------------
        |        shape: (7,), low: -inf, high: inf          | shape:(2, 50), low, high: y좌표제한 |
        """
        self.observation_space = gym.spaces.Box(
           low = -np.inf,
           high = np.inf,
           shape = (7+2*50, ),
           dtype = np.float32
           )

        self.vehicle_state = None
        self.dt = 0.03
        self.border_left = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_left.tolist())
        self.border_right = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_right.tolist())

        self._X_start = self._track.X[0]
        self._X_end = self._track.X[-1]

        self._cnt_mpc_status = 0
        self.before_X = None

        ec_data = np.load("environment/track/data/ec_data.npz")
        self.reference_ec = CubicSpline(ec_data["x"], ec_data["ec"])

        # Just for checking....
        self.total_reward = 0
        self.ep_len = 0

        print("Finishing Initialization")


    def reset(self, seed=None):
        self.vehicle_state, info = self.slalom_env.reset(seed=seed, return_info=True)
        # observation = self.observation_space.sample()
        # observation["vehicle_state"] = np.append(self.vehicle_state.astype(np.float32)[1:-1], np.array([self.dt]))
        # observation["barrier"] = self._get_barrier().astype(np.float32)
        observation = np.append(self.vehicle_state.astype(np.float32)[1:-1], np.array([self.dt]))
        observation = np.append(observation, self._get_barrier().astype(np.float32))

        self.controller = MPC(self._vehicle_model, self._track, self._config)

        self._cnt_mpc_status = 0
        self.before_X = None
        self.dt = 0.03

        self.total_reward = 0
        self.ep_len = 0

        return observation, info


    def step(self, action):
        """
        action: [ddt, array of ecref]
        state: [X, Y, psi,]
        """

        self.dt = self.dt + 0.001*action[0] # should be between 0.01 ~ 0.05
        # ecref = np.zeros((6,)) 
        ecref = np.append(np.array([self.vehicle_state[1]]), action[1:]) # 현 위치의 ec =0 추가

        # Solve MPC
        control, feasible = self.controller.solve(self.vehicle_state, self.dt, ecref)
        # Step with environment
        self.vehicle_state, _, done, truncated, info = self.slalom_env.step(control, self.dt, self.controller.trajectories)
        
        # Check if MPC is feasible
        if self.controller.status != 0:
            self._cnt_mpc_status += 1
        else:
            self._cnt_mpc_status = 0
        if self._cnt_mpc_status >= 10 or feasible:
            done = True
        if self.dt < 0:
            done = True

        observation = np.append(self.vehicle_state.astype(np.float32)[1:-1], np.array([self.dt]))
        observation = np.append(observation, self._get_barrier().astype(np.float32))
        reward = self.get_reward(done, truncated, feasible)
        self.total_reward += reward
        self.ep_len += 1

        self.before_X = self.vehicle_state[0]
        # print(reward)

        if done or truncated:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("reward:", self.total_reward, "episode length:", self.ep_len)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        return observation, reward, done, truncated, info # - 1/100*self.cost


    def get_reward(self, done, truncated, feasible):
        reward = 0

        # 충돌시 penalty
        if self.slalom_env.collided_cone is not None:
            reward += -10

        # minimize ec
        reward += abs(self.reference_ec(self.vehicle_state[0])) - abs(self.vehicle_state[1])

        # when MPC gets unfeasible
        if self._cnt_mpc_status >= 10 or feasible:
            reward += -5
            # print("feasible:", -5)
        
        if self.dt < 0.001:
            reward += -10

        return reward
        

    def render(self):
        self.slalom_env.render()
        

    def _get_barrier(self):
        X = self.vehicle_state[0]
        barrier = np.ndarray((2, 50))

        for i in range(int(self.h / 2)):
            if X + self.h > self._X_end:
                barrier[0, i] = self.border_left(self._X_end)
                barrier[1, i] = self.border_right(self._X_end)
            else:
                barrier[0, i] = self.border_left(X + 2*i)
                barrier[1, i] = self.border_right(X + 2*i)
                
        return barrier