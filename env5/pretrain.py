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
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from callbacks import getBestRewardCallback, logDir, rmsLogging
from stable_baselines3.common.vec_env import VecMonitor
from gym import spaces
from stable_baselines.common.base_class import _UnvecWrapper
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
                         n_episodes=100, image_folder='recorded_images'):

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
        if isinstance(env, VecFrameStack) and env.n_stack == 4:
            # assert env.n_stack < 5, "The current data recorder does no support"\
            #                          "VecFrameStack with n_stack > 4"
            image_ext = 'png'

        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

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

    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]
    while ep_idx < n_episodes:
        obs_ = obs[0] if is_vec_env else obs
        observations.append(obs_)
        action = model(obs)

        obs, reward, done, _ = env.step(action)

        # Use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None

            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1
def make_env(rank, road_type, seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"

    num_proc = 2
    prefix = 'pretrain'

    env = make_env(0, road_type=road_type)

    input("Program Start.\n")

    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=os.path.join(f"tensorboard/{prefix}"))
    generate_expert_traj(model, 'expert_carmaker', n_timesteps=int(1e5), n_episodes=10)



if __name__ == '__main__':
    main()
