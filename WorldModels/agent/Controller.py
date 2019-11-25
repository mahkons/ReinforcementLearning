import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

HIDDEN_ACTOR_SIZE = 256
HIDDEN_CRITIC_SIZE = 256
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3

class Actor(nn.Module):
    def __init__(self, state_sz, action_sz):
        super(Actor, self).__init__()              
        self.fc1 = nn.Linear(state_sz, HIDDEN_ACTOR_SIZE)
        self.fc2 = nn.Linear(HIDDEN_ACTOR_SIZE, action_sz)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_sz, action_sz):
        super(Critic, self).__init__()              
        self.fc1 = nn.Linear(state_sz + action_sz, HIDDEN_CRITIC_SIZE)
        self.fc2 = nn.Linear(HIDDEN_CRITIC_SIZE, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = self.fc2(x)
        return x

class ControllerAC:
    def __init__(self, env, state_sz, action_sz):
        self.env = env
        self.state_sz = state_sz
        self.action_sz = action_sz

        self.actor = Actor(state_sz, action_sz)
        self.critic = Critic(state_sz, action_sz)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.steps_done = 0

    def select_action(self, state):
        return self.env.action_space.sample()
        with torch.no_grad():
            return self.actor.forward(state)

