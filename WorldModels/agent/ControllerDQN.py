import random
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from agent.ReplayBuffer import Transition

HIDDEN_DQN = 32
DQN_N_ATOMS = 3
BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 500
DQN_LR = 1e-3

class DQN(nn.Module):
    def __init__(self, state_sz, action_sz, n_atoms):
        super(DQN, self).__init__()              
        self.fc1 = nn.Linear(state_sz, HIDDEN_DQN)
        self.fc2 = nn.Linear(HIDDEN_DQN, n_atoms ** action_sz)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

class ControllerDQN:
    def __init__(self, env, state_sz, action_sz, device, load_net=None):
        self.env = env
        self.state_sz = state_sz
        self.action_sz = action_sz
        self.device = device

        self.steps_done = 0
        self.net = DQN(state_sz, action_sz, DQN_N_ATOMS).to(device)
        self.target_net = DQN(state_sz, action_sz, DQN_N_ATOMS).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=DQN_LR)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                best_action = self.net(state).max(1)[1]
                return best_action.item()
        else:
            return random.randint(0, DQN_N_ATOMS ** self.action_sz - 1)

    def optimize(self):
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        
        batch_state = torch.cat(batch.state)
        batch_action = torch.cat(batch.action)
        batch_reward = torch.cat(batch.reward)
        batch_next_state = torch.cat(batch.next_state)

        state_action_values = self.net(batch_state).gather(1, batch_action)
        
        with torch.no_grad():
            next_action = self.net(batch_next_state).max(1)[1].unsqueeze(1)
            next_values = self.target_net(batch_next_state).gather(1, next_action).squeeze(1)
        
        expected_state_action_values = (next_values * GAMMA) + batch_reward

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_model(self, path):
        self.net.to(torch.device('cpu'))
        self.target_net.to(torch.device('cpu'))
        self.net.load_state_dict(torch.load(path, map_location='cpu'))
        self.target_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.net.to(self.device)
        self.target_net.to(self.device)

    def save_model(self, path):
        self.net.to(torch.device('cpu'))
        self.target_net.to(torch.device('cpu'))
        torch.save(self.net.state_dict(), path)
        self.net.to(self.device)
        self.target_net.to(self.device)
