from carmaker_env_low import CarMakerEnv
import numpy as np
from stable_baselines3 import SAC
import os

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':
    directory = r"models\DLC\rws_low"
    files = list_files(directory)
    filtered_files = [f for f in files if f.endswith('.pkl')]
    filtered_files = [f for f in filtered_files if f != 'model.pkl']
    filtered_files.sort(key=lambda x: int(x.split('_')[0]))
    filtered_files2 = [filtered_files[i] for i in range(len(filtered_files)) if (i > 10 and i % 40 == 0) or (i < 30 and i % 3 == 0)]
    print(np.size(filtered_files2))
    road_type = "DLC"
    data_name = 'rl'
    comment = "rws"
    prefix = data_name + "_" + comment

    env = CarMakerEnv(road_type=road_type, port=10000 + 0, env_num=0)
    action_lst = []
    reward_lst = []
    info_lst = []
    done_lst = []
    observation_lst = []
    for_next_observation_lst = []
    next_observation_lst = []
    for idx, i in enumerate(filtered_files2):
        model = SAC.load(f"{directory}\{i}", env=env)
        print(f"Model {i} loaded.")
        obs = env.reset()[0]
        tmp = 0
        while True:
            action = model.predict(obs)
            action = action[0]
            obs, reward, done, _, info = env.step(action)
            action_lst.append(action)
            observation_lst.append(obs)
            for_next_observation_lst.append(obs)
            reward_lst.append(reward)
            done_lst.append(done)
            info_lst.append(info)

            if done:
                # 마지막 데이터는 직전 데이터로
                observation_len = len(for_next_observation_lst)
                for idx2 in range(observation_len - 1):
                    next_observation_lst.append(observation_lst[idx2 + 1])
                next_observation_lst.append(observation_lst[-1])
                for_next_observation_lst = []
                print(f"{idx} obs shape : {np.shape(observation_lst)}")
                break


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

    print("Data Saved.")