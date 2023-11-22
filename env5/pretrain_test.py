"""
학습 코드 예제
1. 카메이커 연동 환경을 불러온다
    1-1. 여러 대의 카메이커를 실행하기 위해 SubprocVecEnv를 이용하여 멀티프로세싱이 가능한 환경 로드
2. 학습에 사용할 RL 모델(e.g. PPO)을 불러온다.
3. 학습을 진행한다. x
    3-1. total_timesteps 수를 변화시켜 충분히 학습하도록 한다.
4. 학습이 완료된 후 웨이트 파일(e.g. model.pkl)을 저장한다.
"""
import numpy as np
import warnings
from carmaker_env_low import CarMakerEnv
from stable_baselines3 import SAC, PPO
from typing import Dict
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from callbacks import getBestRewardCallback, logDir, rmsLogging
from stable_baselines3.common.vec_env import VecMonitor
from gym import spaces
import os
import torch
import logging
from datetime import datetime

logging.basicConfig(filename='Log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
custom_logger = logging.getLogger('customLogger')
custom_logger.propagate = False
handler = logging.FileHandler('Log.txt')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
custom_logger.addHandler(handler)
custom_logger.setLevel(logging.INFO)

# GPU를 사용할 수 있는지 확인합니다.
if torch.cuda.is_available():
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device("cuda:" + str(device_id))
    print(f"Using GPU device ID {device_id}.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

def generate_expert(model, save_path=None, env=None, n_timesteps=0,
                         n_episodes=5, image_folder='recorded_images'):
    print('here')

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
                    and obs_space.dtype == np.uint8
    if record_images and save_path is None:
        warnings.warn("Observations are images but no save path was specified, so will save in numpy archive; "
                      "this can lead to higher memory usage.")
        record_images = False

    if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved in the numpy archive "
                      "which can lead to high memory usage".format(obs_space.shape))

    image_ext = 'jpg'
    if record_images:
        # We save images as jpg or png, that have only 3/4 color channels

        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    model.learn(n_timesteps)
    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None


    while ep_idx < n_episodes:
        obs_ = obs
        observations.append(obs_)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(action)

        # Use only first env

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            obs = env.reset()

            print(f"New observation after reset: {obs}")

            # 에피소드별 총 보상 기록
            print(f"Episode {ep_idx} finished. Total reward: {reward_sum}")

            # 현재 에피소드의 마지막 액션 및 상태 기록
            print(f"Last action: {action}, Last observation: {obs_}")

            # Reset the state in case of a recurrent policy
            state = None

            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1
            print(f"observation: {np.shape(observations)}, reward: {np.shape(actions)}")
            if ep_idx == n_episodes - 2:
                if isinstance(env.observation_space, spaces.Box) and not record_images:
                    observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
                elif isinstance(env.observation_space, spaces.Discrete):
                    observations = np.array(observations).reshape((-1, 1))
                elif record_images:
                    observations = np.array(observations)

                if isinstance(env.action_space, spaces.Box):
                    actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
                elif isinstance(env.action_space, spaces.Discrete):
                    actions = np.array(actions).reshape((-1, 1))

                rewards = np.array(rewards)
                episode_starts = np.array(episode_starts[:-1])

                assert len(observations) == len(actions)

                numpy_dict = {
                    'actions': actions,
                    'obs': observations,
                    'rewards': rewards,
                    'episode_returns': episode_returns,
                    'episode_starts': episode_starts
                }  # type: Dict[str, np.ndarray]

                for key, val in numpy_dict.items():
                    print(key, val.shape)

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if save_path is not None:
                    np.savez(save_path, **numpy_dict)

                env.close()
                print("Saved")

    return numpy_dict

def make_env(rank, road_type, seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, simul_path='test_IPG', port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"

    prefix = 'pretrain'

    env = make_env(0, road_type=road_type)()

    input("Program Start.\n")

    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=os.path.join(f"tensorboard/{prefix}"))
    generate_expert(model, f'{road_type}/expert_carmaker', env, n_timesteps=int(1e5), n_episodes=100)



if __name__ == '__main__':
    main()
