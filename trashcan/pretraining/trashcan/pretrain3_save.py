"""
학습 후 테스트하는 코드 예제
1. 카메이커 연동 환경을 불러온다
2. 학습에 사용한 RL 모델(e.g. PPO)에 학습된 웨이트 파일(e.g. model.pkl)을 로드한다.
3. 테스트를 수행한다.
"""

from carmaker_env_low_pretrain import CarMakerEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines import SAC

if __name__ == '__main__':
    road_type = "DLC"
    data_name = 'IPG'
    comment = "rws"
    prefix = data_name + "_" + comment


    env = CarMakerEnv(port=9999, road_type=road_type, simul_path='pythonCtrl_pretrain', use_low=True, check=0)
#    model = SAC.load(f"best_model/1519999_Check_model.pkl", env=env)
    model = SAC.load(f"best_model/DLC_best_model.pkl", env=env)
    print("Model loaded.")

    buffer_size = 10 * 10000
    obs = env.reset()
    action_lst = []
    reward_lst = []
    obs_lst = []
    info_lst = []
    done_lst = []
    step_num = 0
    done2 = False
    while True:

        action = model.predict(obs)
        obs, reward, done, info = env.step(action[0])

        action_lst.append(info["action"])
        obs_lst.append(obs)
        reward_lst.append(reward)
        done_lst.append(done)
        info_lst.append(info)
        if done:
            print(f"Step Number: {step_num}")
            obs = env.reset()
            break

    expert_obs_lst = []
    expert_action_lst = []
    expert_reward_lst = []
    expert_done_lst = []
    expert_info_lst = []

    np.savez('expert_data.zip',
             buffer_size=buffer_size,
             observations=np.array(obs_lst),
             actions=np.array(action_lst),
             rewards=np.array(reward_lst),
             dones=np.array(done_lst),
             infos=info_lst)
