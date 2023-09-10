"""
학습 후 테스트하는 코드 예제
1. 카메이커 연동 환경을 불러온다
2. 학습에 사용한 RL 모델(e.g. PPO)에 학습된 웨이트 파일(e.g. model.pkl)을 로드한다.
3. 테스트를 수행한다.
"""

from carmaker_env13 import CarMakerEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

if __name__ == '__main__':
    data_name = 'RL_ISO3888'
    env = CarMakerEnv(host='127.0.0.1', port=9999)
    model = SAC.load("env13_best_model.pkl", env=env)
    print("Model loaded.")

    obs = env.reset()
    action1 = []
    reward_arr=[]
    info_lst = []

    tmp=0
    while True:
        tmp+=1
        action = model.predict(obs)
        obs, reward, done, info = env.step(action[0])
        info_lst.append(info)
        if tmp % 5 ==0:
            action1.append(action[0])
            reward_arr.append(reward)

        if done:
            df1 = pd.DataFrame(data=reward_arr)
            df1.to_csv('{}_reward.csv'.format(data_name))
            df3 = pd.DataFrame(data=info_lst)
            df3.to_csv('{}_info.csv'.format(data_name), index=False)
            df4 = pd.DataFrame(data=action1)
            df4.to_csv('{}_action.csv'.format(data_name), index=False)
            print("Episode Finished.")
            break
