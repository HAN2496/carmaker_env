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
    data_name = 'pretrain'
    buffer_size = 100
    pretrain_steps = 500
    learn_steps = 100
    comment = f"buffer{buffer_size}k_pretrain{pretrain_steps}k_learn{learn_steps}k"
    buffer_size = buffer_size * 1000
    pretrain_steps = pretrain_steps * 1000
    learn_steps = learn_steps * 1000
    prefix = data_name + "_" + comment


    env = CarMakerEnv(port=9999, road_type=road_type, use_carmaker=True, env_num=0)
    model = SAC.load(f"models/DLC/{comment}/25099_best_model.pkl", env=env)
    print("Model loaded.")

    obs = env.reset()[0]

    action_lst = []
    reward_lst=[]
    info_lst = []
    while True:
        action = model.predict(obs)
        action = action[0]
        obs, reward, done, _, info = env.step(action)
        info_lst.append(info)
        action_lst.append(action)
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
