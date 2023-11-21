import os
from stable_baselines3 import PPO
from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common import save_traj
import os
import torch

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

num_proc = 2
road_type = "YourRoadType"
env = SubprocVecEnv([make_env(i, road_type=road_type) for i in range(num_proc)])

expert_model = PPO('MlpPolicy', env, verbose=1)
expert_model.learn(total_timesteps=10000)


# Evaluate the policy and save the trajectories
eval_env = make_env(0, road_type=road_type)()
save_path = "./expert_data"
os.makedirs(save_path, exist_ok=True)
save_traj(eval_env, expert_model, save_path, n_eval_episodes=100)