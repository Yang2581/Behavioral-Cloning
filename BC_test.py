import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

class BehavioralCloning(nn.Module):
    def __init__(self, n_states, n_actions, lr):
        super(BehavioralCloning, self).__init__()

        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

        self.lr = lr

        self.lossfn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.tanh(x)

        return output

    def choose_action(self, x):
        x = torch.Tensor(x)
        x = x.to(torch.float32)
        return self.forward(x)

if __name__ == "__main__":
    n_states = 3
    n_actions = 1
    lr = 0.001
    model = BehavioralCloning(n_states, n_actions, lr)
    model.float()
    model.load_state_dict(torch.load('weight.pt'))

#################### test our model #####################
    # env = gym.make('Pendulum-v1')
    env = gym.make('Pendulum-v1', render_mode = 'human', new_step_api = True)
    n_games = 50

    for i in range(n_games):
        observation = env.reset()
        # done = False
        score = 0
        terminated = False
        truncated = False
        done = terminated | truncated

        while not done:
            # env.render()
            action = model.choose_action(observation)
            # observation_, reward, done, info = env.step(action.detach().numpy())
            observation_, reward, terminated, truncated, info = env.step(action.detach().numpy())
            observation = observation_
            score += reward
            done = terminated | truncated
        print('episode ', i, 'score %.1f' % score)