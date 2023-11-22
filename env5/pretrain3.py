import gym
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_expert_dataset

# 환경 생성
env = gym.make('YourEnvName')
env = DummyVecEnv([lambda: env])  # 벡터화된 환경

# 전문가 데이터 로드
dataset = get_expert_dataset('path_to_expert_data.npz')

# Behavior Cloning 모델 생성 및 학습
bc_model = sb3.BC(policy='MlpPolicy', env=env, dataset=dataset)
bc_model.learn(total_timesteps=10000)

# SAC 모델 생성 및 전문가 정책으로 초기화
sac_model = sb3.SAC('MlpPolicy', env, verbose=1)
sac_model.policy.load_state_dict(bc_model.policy.state_dict())

# SAC 모델 학습
sac_model.learn(total_timesteps=100000)

# 성능 평가
mean_reward, std_reward = evaluate_policy(sac_model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")
