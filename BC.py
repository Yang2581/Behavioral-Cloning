'''
author: Xiaoyang
time: 11 Sep. 2021 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import gym

torch.set_default_tensor_type(torch.FloatTensor)
class MyDataset(Dataset):
    def __init__(self, file_path):
        dataset = np.load(file_path, allow_pickle=True).item()

        self.observations = torch.from_numpy(dataset['observations'])
        self.actions = torch.from_numpy(dataset['actions'])

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

    def __len__(self):
        return len(self.observations)

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
        output = F.tanh(x)

        return output

    def choose_action(self, x):
        x = torch.Tensor(x)
        x = x.to(torch.float32)
        return self.forward(x)

if __name__ == "__main__":
    file_path = 'datasets10000.npy'
    n_states = 3
    n_actions = 1
    lr = 0.001
    model = BehavioralCloning(n_states, n_actions, lr)
    model.float()
    mydataset = MyDataset(file_path)
    train_loader = DataLoader(dataset=mydataset, batch_size=16, shuffle=True)

    for epoch in range(4):
        for i, data in enumerate(tqdm(train_loader)):
            states, actions = data
            states = states.to(torch.float32)
            actions = actions.to(torch.float32)
            model.optimizer.zero_grad()
            pre_actions = model.forward(states)
            loss = model.lossfn(pre_actions, actions)
            loss.backward()
            model.optimizer.step()

            if i % 10 == 9:
                print('epoch: {} , i: {}, loss: {}'.format(epoch, i, loss))
    print('finished training!')
    save_path = 'weight.pt'
    torch.save(model.state_dict(), save_path)
    # print(model.choose_action(0.1))