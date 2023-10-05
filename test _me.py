import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MetaController(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_dim=256):
        super(MetaController, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def select_goal(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            goal = self.policy_net(state)
        return goal.numpy().flatten()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class Controller(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(Controller, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def select_action(self, state, goal):
        input_tensor = torch.FloatTensor(np.concatenate([state, goal])).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(input_tensor)
        return action.numpy().flatten()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class HSAC:
    def __init__(self, state_dim, goal_dim, action_dim):
        self.meta_controller = MetaController(state_dim, goal_dim)
        self.controller = Controller(state_dim + goal_dim, action_dim)

    def train(self, env, episodes=1000):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                env.render()
                goal = self.meta_controller.select_goal(state)
                action = self.controller.select_action(state, goal)
                next_state, reward, done, _ = env.step(action)

                # Here, normally, you'd add the experience to some replay buffer
                # and sample from that buffer to update the controllers.

                # Example of a simple update (not considering discount factor, target networks, etc.)
                # You'd want to replace this with proper SAC updates
                meta_loss = F.mse_loss(torch.FloatTensor(next_state), torch.FloatTensor(goal))
                self.meta_controller.update(meta_loss)

                controller_loss = -torch.FloatTensor([reward])  # minimizing negative reward
                self.controller.update(controller_loss)

                state = next_state
        env.close()


if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = state_dim  # Assuming goal_dim is same as state_dim

    agent = HSAC(state_dim, goal_dim, action_dim)
    agent.train(env)

