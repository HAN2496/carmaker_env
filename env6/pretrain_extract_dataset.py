from carmaker_env_low import CarMakerEnv
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

if __name__ == '__main__':
    road_type = "DLC"
    data_name = 'rl'
    comment = "rws"
    prefix = data_name + "_" + comment

    env = CarMakerEnv(road_type=road_type, port=10000 + 0, env_num=0)
    model = SAC.load(f"best_model/DLC_best_model.pkl", env=env)
    print("Model loaded.")
    obs = env.reset()[0]


    action_lst = []
    reward_lst = []
    info_lst = []
    done_lst = []
    observation_lst = []
    next_observation_lst = []
    while True:
        action = model.predict(obs)
        action = action[0]
        obs, reward, done, _, info = env.step(action)
        action_lst.append(action)
        observation_lst.append(obs)
        reward_lst.append(reward)
        done_lst.append(done)
        info_lst.append(info)
        if done:
            df1 = pd.DataFrame(data=reward_lst)
            df1.to_csv(f'datafiles/{road_type}/{prefix}_reward.csv')
            df2 = pd.DataFrame(data=observation_lst)
            df2.to_csv(f'datafiles/{road_type}/{prefix}_observation.csv', index=False)
            df3 = pd.DataFrame(data=info_lst)
            df3.to_csv(f'datafiles/{road_type}/{prefix}_info.csv', index=False)
            df4 = pd.DataFrame(data=action_lst)
            df4.to_csv(f'datafiles/{road_type}/{prefix}_action.csv', index=False)
            df5 = pd.DataFrame(data=done_lst)
            df5.to_csv(f'datafiles/{road_type}/{prefix}_done.csv', index=False)
            print("Episode Finished. Data saved.")
            break

    #마지막 데이터는 직전 데이터로
    observation_len = len(observation_lst)
    for idx in range(observation_len - 1):
        next_observation_lst.append(observation_lst[idx + 1])
    next_observation_lst.append(observation_lst[-1])

    #잘 들어갔나 확인 (위에서 observation_lst[-1]로 해버려서 혹시 모르니)
    print(np.shape(observation_lst))
    print(np.shape(next_observation_lst))
    print(observation_lst[-1])
    print(next_observation_lst[-1])
    
    #데이터 저장
    np.savez(f'expert_data.npz',
             observations=np.array(observation_lst),
             next_observations=np.array(next_observation_lst),
             actions=np.array(action_lst),
             rewards=np.array(reward_lst),
             dones=np.array(done_lst),
             infos=np.array(info_lst))
