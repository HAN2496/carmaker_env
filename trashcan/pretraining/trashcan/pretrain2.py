import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer
from carmaker_env_low import CarMakerEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank,road_type,  simul_path='pythonCtrl_JX1', seed=0):

    def _init():
        env = CarMakerEnv(road_type=road_type, simul_path=simul_path, port=10000 + rank, check=rank)  # 모니터 같은거 씌워줘야 할거임
        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init

def main():
    road_type = "DLC"

    prefix = 'pretrain'

    env = make_env(0, road_type=road_type, simul_path='test_IPG')()

    print("Program Start.\n")

    expert_model = SAC('MlpPolicy', env, verbose=1)
    expert_model.learn(total_timesteps=100)

    print("Expert learning finished. Expert data will be collected.")
    # 전문가 데이터 수집
    expert_actions = []
    expert_observations = []
    expert_next_observations = []
    expert_reward = []
    expert_done = []
    expert_info = []
    obs = env.reset()
    buffer_size = 100
    for _ in range(buffer_size):
        action = expert_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action[0])
        expert_observations.append(obs)
        expert_reward.append(reward)
        expert_done.append(done)
        expert_actions.append(action[0])
        expert_info.append(info)
        if done:
            obs = env.reset()
            break

    print(f"action: {np.shape(expert_actions)} / obs: {np.shape(expert_observations)} / info: {np.shape(expert_info)}")
    print("Collecting Expert data stopped.")
    print("$$$$")
    print(expert_info[0])
    print(expert_info[1])
    env = make_env(0, road_type=road_type)()
    # 모방 학습을 위한 SAC 모델
    model = SAC("MlpPolicy", env, verbose=1)

    # 전문가 데이터를 리플레이 버퍼에 추가
    replay_buffer = ReplayBuffer(buffer_size=1000, observation_space=env.observation_space,
                                 action_space=env.action_space)
    print('-----')
    for idx, (obs, action, reward, done, info) in enumerate(zip(expert_observations, expert_actions, expert_reward, expert_done, expert_info)):
        next_obs = expert_observations[idx + 1] if idx < len(expert_observations) - 1 else None
        print(f"obs shape: {np.shape(obs)} / obs type: {type(obs)}")
        print(f"info shape: {np.shape(info)} / info type: {type(info)}")
        replay_buffer.add(obs, next_obs, action, reward, done, [info])

        # 사전 학습된 데이터로 모델 초기화
    model.replay_buffer = replay_buffer

    # 모델 훈련
    model.learn(total_timesteps=10000)

    # 사전 학습된 데이터로 모델 초기화
    model.replay_buffer = replay_buffer

    # 모델 훈련
    model.learn(total_timesteps=100)

if __name__ == '__main__':
    main()