from controller import Controller
from meta_controller import MetaController
import torch
import torch.nn.functional as F
import gym

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

