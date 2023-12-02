"""
학습 후 테스트하는 코드 예제
1. 카메이커 연동 환경을 불러온다
2. 학습에 사용한 RL 모델(e.g. PPO)에 학습된 웨이트 파일(e.g. model.pkl)을 로드한다.
3. 테스트를 수행한다.
"""

from carmaker_env_low import CarMakerEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

if __name__ == '__main__':
    road_type = "DLC"
    data_name = 'RL'
    comment = "rws"
    prefix = data_name + "_" + comment


    env = CarMakerEnv(port=9999, road_type=road_type, use_carmaker=True, env_num=0)
#    model = SAC.load(f"best_model/1519999_Check_model.pkl", env=env)
    model = SAC.load(f"best_model/DLC_best_model.pkl", env=env)
    print("Model loaded.")

    obs = env.reset()
    action_lst = []
    reward_lst=[]
    info_lst = []
    while True:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action[0])
        info_lst.append(info)
        action_lst.append(action[0])
        reward_lst.append(reward)
        if done:
            df1 = pd.DataFrame(data=reward_lst)
            df1.to_csv(f'datafiles/{road_type}/{prefix}_reward.csv')
            df3 = pd.DataFrame(data=info_lst)
            df3.to_csv(f'datafiles/{road_type}/{prefix}_info.csv', index=False)
            df4 = pd.DataFrame(data=action_lst)
            df4.to_csv(f'datafiles/{road_type}/{prefix}_action.csv', index=False)
            print("Episode Finished. Data saved.")
            break
