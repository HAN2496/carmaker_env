import numpy as np
import gymnasium as gym
from casadi import interpolant
from scipy.interpolate import CubicSpline

from carmaker_env13 import CarMakerEnv
from environment.config import Config
from environment.vehicle_model import VehicleModel
from environment.track import Track
from mpc import MPC

class RLCarMakerEnv(gym.Env):
    """
    RL environment for training with CarMaker-Simulink.
    """

    def __init__(self, port, matlab_path, simul_path, save_data, render_mode=None):
        super(RLCarMakerEnv, self).__init__()

        self._config = Config()
        self._vehicle_model = VehicleModel(self._config)
        self._track = Track()
        self.controller = MPC(self._vehicle_model, self._track, self._config)
        self.carmaker_env = CarMakerEnv(host='127.0.0.1', port=port, matlab_path=matlab_path, simul_path=simul_path, save_data=save_data)
        
        self.h = self._config.rl.h

        """
        Actions : (6,)
        |     ddt     |  5 points of reference ec  |
        ---------------------------------------------
        | shape: (1,) |         shape: (5,)        |
        """
        self.action_space = gym.spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (6,),
            dtype = np.float32
        )

        """
        Observation : (112,)
        |  vehicle state: Y, psi, vx, vy, omega, delta, dt, 5 ec         |       barrier for length h       |
        ------------------------------------------------------------------------
        |              shape: (7,), low: -inf, high: inf             | shape:(2, 50), low, high: y좌표제한 |
        """
        self.observation_space = gym.spaces.Box(
           low = -np.inf,
           high = np.inf,
           shape = (12+2*50, ),
           dtype = np.float32
           )

        self.vehicle_state = None
        self.dt = 0.02
        self.ecref = np.zeros((6,0)) # first ecref is always the current ec of the vehicle
        self.border_left = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_left.tolist())
        self.border_right = interpolant("bound_l_s", "bspline", [self._track.X.tolist()], self._track.border_right.tolist())
        self.control = np.zeros((1,))

        self._X_start = self._track.X[0]
        self._X_end = self._track.X[-1]

        self._cnt_mpc_status = 0
        self.sim_started = False
        self.before_X = np.array([0.0])

        # Load ec reference for reward
        ec_data = np.load("environment/track/data/ec_data.npz")
        x = np.append(ec_data["x"], np.linspace(401, 501, 101))
        ec = np.append(ec_data["ec"], np.zeros((101,)))
        self.reference_ec = CubicSpline(x, ec)

        # For debugging
        self.total_reward = 0
        self.ep_len = 0

        print("Finishing Initialization")


    def reset(self, seed=None):
        print("\nStarting reset\n")
        self.vehicle_state = self.carmaker_env.reset()
        self.dt = 0.02
        self.ecref = np.zeros((6,))
        self.ecref[0] = self.vehicle_state[1]
        self.control = np.zeros((1,))

        observation = np.append(self._get_normalized_state(), np.array([self.dt]))
        observation = np.append(observation, self.ecref[1:])
        observation = np.append(observation, self._get_barrier().astype(np.float32))

        self.controller = MPC(self._vehicle_model, self._track, self._config)
        self._cnt_mpc_status = 0
        self.sim_started = False
        self.before_X = self.vehicle_state[0]

        self.total_reward = 0
        self.ep_len = 0

        return observation.astype(np.float32), {}


    def step(self, action):
        """
        action: [ddt, array of ecref]. shape:(6,)
        """
        # Update param
        self.dt = self.dt + 0.001*action[0] # should be between 0.01 ~ 0.05
        ecref = np.append(np.array([self.vehicle_state[1]]), action[1:]) # 현 위치의 ec =0 추가
        
        # Init
        done = False
        simul_done = False
        info = {}

        # Solve MPC
        self.control, feasible = self.controller.solve(self.vehicle_state, self.dt, ecref)

        # Step with environment depending on dt
        for _ in range(int(self.dt/0.001)):
            self.vehicle_state, _, simul_done, info = self.carmaker_env.step(self.control)
            if simul_done:
                break

        # Check if MPC is feasible
        if self.sim_started:
            if self.controller.status != 0:
                self._cnt_mpc_status += 1
            else:
                self._cnt_mpc_status = 0
        else: 
            if self.controller.status == 0:
                self.sim_started = True
                print("\nSim Started\n")
        
        # Check for done
        if self._cnt_mpc_status >= 10 or feasible:
            print("DONE: MPC issue")
            done = True
        if self.dt < 0.001:
            print("DONE: dt<0")
            done = True

        # Get observation
        observation = np.append(self._get_normalized_state(), np.array([self.dt]))
        observation = np.append(observation, self.ecref[1:])
        observation = np.append(observation, self._get_barrier().astype(np.float32))
        
        # Get reward
        reward = self.get_reward(feasible=feasible)
        
        self.total_reward += reward
        self.ep_len += 1

        if done:
            print("\nForce quiting")
            sim_control = np.array([0.0])
            while True:
                sim_state, _, simul_done, _ = self.carmaker_env.step(sim_control)
                if sim_state[-2] > 0:
                    sim_control = np.array([-1])
                elif sim_state[-2] == 0:
                    sim_control = np.array([0.0])
                else:
                    sim_control = np.array([1])
                # print("delta:", sim_state[-2], "sim_control:", sim_control)
                
                if simul_done:
                    break

        done = done or simul_done
        if done:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("reward:", self.total_reward, "episode length:", self.ep_len)
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        
        return observation.astype(np.float32), reward, done, False, info # - 1/100*self.cost


    def get_reward(self, feasible):
        reward = 0

        # 충돌시 penalty
        if self.check_collision():
            reward += -20

        # maximize theta
        reward += 0.2*(self.vehicle_state[0] - self.before_X)

        # minimize ec
        ref_ec = self.reference_ec(self.vehicle_state[0])
        if ref_ec*self.vehicle_state[1] > 0: # 같은 부호
            reward += np.sign(ref_ec)*(ref_ec - self.vehicle_state[1])
        else: # 다른 부호
            reward += -abs(ref_ec - self.vehicle_state[1])

        # when MPC gets unfeasible
        if self._cnt_mpc_status >= 10 or feasible:
            reward += -20
        
        if self.dt < 0.005:
            reward += -20*(0.001/self.dt)


        return reward
        

    def render(self):
        pass
        
        

    def check_collision(self):
        cone_position, cone_index = self._track.get_nearlest_cone(self.vehicle_state[0], return_cone_index=True)
        cone_position_rel = self._vehicle_model.R(self.vehicle_state[2]).T @ (cone_position - self.vehicle_state[:2])

        if abs(cone_position_rel[0]) < 0.5 * self._vehicle_model.L + self._config.env.cone_radius and abs(cone_position_rel[1]) < 0.5 * self._vehicle_model.W + self._config.env.cone_radius:
            self.collided_cone = (cone_position, cone_index)
            print("Collision occurred")
            return True
        return False


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
    

    def _get_normalized_state(self): 
        # X, Y, psi, vx, vy, omega, delta, tau

        Y = self.vehicle_state[1]
        psi = self.vehicle_state[2]/0.4
        vx = (self.vehicle_state[3] - self._config.env.vref)/0.5
        vy = self.vehicle_state[4]/0.4
        omega = self.vehicle_state[5]/0.5
        delta = self.vehicle_state[-2]/0.15

        return np.array([Y, psi, vx, vy, omega, delta], dtype=np.float32)