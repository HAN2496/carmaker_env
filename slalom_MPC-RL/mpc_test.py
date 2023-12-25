import numpy as np

from environment import SlalomTest, Config
from environment.vehicle_model import VehicleModel
from environment.track import Track
from mpc import MPC

import matplotlib.pyplot as plt

if __name__=="__main__":
    print("hello")

    config = Config()

    vehicle_model = VehicleModel(config)

    track = Track()

    controller = MPC(vehicle_model, track, config)

    env = SlalomTest(render_mode="human")

    state = env.reset()

    ec_list = []
    ec_list.append(state[1])
    x_list = []
    x_list.append(state[0])

    for i in range(4000):
    
        ddelta, _ = controller.solve(state, dt=0.03, ecref=np.zeros((6,)))
        # print("state:", state[0], state[1], state[3], "ddelta: ", ddelta)
        print("x:", state[0], "vx:", state[3])
    
        control = np.array([ddelta, 0.0], dtype=np.float32)    
        state, reward, done, truncated, info = env.step(control, 0.03, controller.trajectories)
        x_list.append(state[0])
        ec_list.append(state[1])
        
        if done or truncated == True:
            print(i)
            plt.figure(2)
            plt.plot(ec_list)
            plt.show()
            np.savez("ec_data.npz",
                ec = ec_list,
                x = x_list
            )
            break

