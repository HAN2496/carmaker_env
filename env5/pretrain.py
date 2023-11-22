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
from stable_baselines.gail import ExpertDataset
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

def pretrain(dataset, n_epochs=10, learning_rate=1e-4,
             adam_epsilon=1e-8, val_interval=None):
    """
    Pretrain a model using behavior cloning:
    supervised learning given an expert dataset.

    NOTE: only Box and Discrete spaces are supported for now.

    :param dataset: (ExpertDataset) Dataset manager
    :param n_epochs: (int) Number of iterations on the training set
    :param learning_rate: (float) Learning rate
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param val_interval: (int) Report training and validation losses every n epochs.
        By default, every 10th of the maximum number of epochs.
    :return: (BaseRLModel) the pretrained model
    """
    continuous_actions = isinstance(self.action_space, gym.spaces.Box)

    assert continuous_actions, 'Only Discrete and Box action spaces are supported'

    # Validate the model every 10% of the total number of iteration
    if val_interval is None:
        # Prevent modulo by zero
        if n_epochs < 10:
            val_interval = 1
        else:
            val_interval = int(n_epochs / 10)

    with self.graph.as_default():
        with tf.variable_scope('pretrain'):
            if continuous_actions:
                obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                loss = tf.reduce_mean(tf.square(actions_ph - deterministic_actions_ph))
            else:
                obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                # so no additional changes is needed in the dataloader
                actions_ph = tf.expand_dims(actions_ph, axis=1)
                one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=actions_logits_ph,
                    labels=tf.stop_gradient(one_hot_actions)
                )
                loss = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
            optim_op = optimizer.minimize(loss, var_list=self.params)

        self.sess.run(tf.global_variables_initializer())

    if self.verbose > 0:
        print("Pretraining with Behavior Cloning...")

    for epoch_idx in range(int(n_epochs)):
        train_loss = 0.0
        # Full pass on the training set
        for _ in range(len(dataset.train_loader)):
            expert_obs, expert_actions = dataset.get_next_batch('train')
            feed_dict = {
                obs_ph: expert_obs,
                actions_ph: expert_actions,
            }
            train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
            train_loss += train_loss_

        train_loss /= len(dataset.train_loader)

        if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
            val_loss = 0.0
            # Full pass on the validation set
            for _ in range(len(dataset.val_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('val')
                val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                    actions_ph: expert_actions})
                val_loss += val_loss_

            val_loss /= len(dataset.val_loader)
            if self.verbose > 0:
                print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                print('Epoch {}'.format(epoch_idx + 1))
                print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                print()
        # Free memory
        del expert_obs, expert_actions
    if self.verbose > 0:
        print("Pretraining done.")
    return self


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
    #generate_expert(model, f'{road_type}/expert_carmaker', env, n_timesteps=int(1e5), n_episodes=100)

    dataset = ExpertDataset(expert_path='DLC/expert_carmaker.npz',
                            traj_limitation=1)
    model.pretrain(dataset)

if __name__ == '__main__':
    main()