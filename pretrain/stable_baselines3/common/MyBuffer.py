from stable_baselines3.common.buffers import ReplayBuffer

class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(CustomReplayBuffer, self).__init__(*args, **kwargs)
        self.infos = []

    def add(self, obs, next_obs, action, reward, done, info):
        super(CustomReplayBuffer, self).add(obs, next_obs, action, reward, done, info)
        self.infos.append(info)
