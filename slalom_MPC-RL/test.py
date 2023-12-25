"""
학습 후 테스트하는 코드 예제
1. 카메이커 연동 환경을 불러온다
2. 학습에 사용한 RL 모델(e.g. PPO)에 학습된 웨이트 파일(e.g. model.pkl)을 로드한다.
3. 테스트를 수행한다.
"""

from RL.rl_env import RLEnv
from RL.rl_carmaker_env import RLCarMakerEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

if __name__ == '__main__':
    
    # Load model
    matlab_path = "/home/seoyeon/mpc_ws/RWS/JX1_102/src_cm4sl"
    simul_path = "pythonCtrl_mpc"
    save_data = False
    env = RLCarMakerEnv(port=10000, matlab_path=matlab_path, simul_path=simul_path, save_data=save_data)
    model = SAC.load("models/model_slalom_1213_last.pkl", env=env)

    obs, _ = env.reset()
    action1 = []
    reward_arr=[]
    info_lst = []
    x_list = []
    ec_list = []
    dt_list = []

    tmp=0
    while True:
        tmp+=1
        # print(tmp, ":",  obs, action)
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        x_list.append(env.vehicle_state[0])
        ec_list.append(env.vehicle_state[1])
        info_lst.append(info)
        reward_arr.append(reward)
        action1.append(env.control)
        dt_list.append(env.dt)
        
        if done or truncated:
            if save_data:
                data_name = "mpcrl_slalom"
                df1 = pd.DataFrame(data=reward_arr)
                df1.to_csv('analysis/{}_reward.csv'.format(data_name))
                df3 = pd.DataFrame(data=info_lst)
                df3.to_csv('analysis/{}_info.csv'.format(data_name), index=False)
                df4 = pd.DataFrame(data=action1)
                df4.to_csv('analysis/{}_action.csv'.format(data_name), index=False)

            print("Episode Finished")
            break
    
    x_list = np.array(x_list)
    ec_list = np.array(ec_list)

    plt.figure(1)
    plt.plot(x_list, ec_list)

    plt.figure(2)
    plt.plot(dt_list)

    plt.show()
