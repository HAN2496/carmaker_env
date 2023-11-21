import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import torch
from carmaker_env_low import CarMakerEnv
from stable_baselines3 import SAC, PPO

def make_env(rank, road_type, seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, simul_path='test_IPG', port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

# GPU를 사용할 수 있는지 확인합니다.
if torch.cuda.is_available():
    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device("cuda:" + str(device_id))
    print(f"Using GPU device ID {device_id}.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")


SEED = 42
env = make_vec_env("seals:seals/CartPole-v0", rng=np.random.default_rng(SEED), n_envs=8, post_wrappers=[lambda env, _: RolloutInfoWrapper(env)])
expert = load_policy("ppo-huggingface", organization="HumanCompatibleAI", env_name="seals-CartPole-v0", venv=env)

rollouts = rollout.rollout(expert, env, rollout.make_sample_until(min_timesteps=None, min_episodes=60), rng=np.random.default_rng(SEED))

learner = PPO(env=env, policy=MlpPolicy, batch_size=64, ent_coef=0.0, learning_rate=0.0004, gamma=0.95, n_epochs=5, seed=SEED)
reward_net = BasicRewardNet(observation_space=env.observation_space, action_space=env.action_space, normalize_input_layer=RunningNorm)

gail_trainer = GAIL(demonstrations=rollouts, demo_batch_size=1024, gen_replay_buffer_capacity=512, n_disc_updates_per_round=8, venv=env, gen_algo=learner, reward_net=reward_net)
gail_trainer.train(20000)

learner_rewards_before_training, _ = evaluate_policy(learner, env, 100, return_episode_rewards=True)
learner_rewards_after_training, _ = evaluate_policy(learner, env, 100, return_episode_rewards=True)
print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))
