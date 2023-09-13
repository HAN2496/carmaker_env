import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
import os
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
# Callback for plotting
def plotting_callback(_locals, _globals):
    global episode_rewards
    # Get the monitor's data
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(x) > 0:
        episode_rewards.append(y[-1])
    if len(episode_rewards) % 10 == 0:  # plot every 10 episodes
        plt.plot(episode_rewards)
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('Training Progress')
        plt.pause(0.01)
    return True

# Initialize the environment and PPO agent
env = Monitor(gym.make('CartPole-v1'), filename=log_dir)
model = PPO("MlpPolicy", env, verbose=1)

# Main training loop with plotting callback
episode_rewards = []
model.learn(total_timesteps=50000, callback=plotting_callback)

# After training, display the progress
plt.show()
