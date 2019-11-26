import numpy as np
import gym
import math
import random
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_

from agent.ReplayBuffer import ReplayMemory, Transition
from worldmodel.mdnrnn.MDNRNN import detach

DQN_N_ATOMS = 3

class Agent:
    def __init__(self, env, rnn, vae, controller, device, mem_size=1000000, z_size=32):
        self.env = env
        self.vae = vae
        self.rnn = rnn

        self.memory = ReplayMemory(mem_size)
        self.controller = controller
        self.controller.memory = self.memory

        self.device = device
        self.action_sz = self.controller.action_sz
        self.z_size = z_size

    def resize_obs(self, obs):
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(64),
            T.ToTensor(),
        ])
        return transform(obs).to(self.device).unsqueeze(0)

    def rollout(self, show=False):
        obs = self.resize_obs(self.env.reset())
        with torch.no_grad():
            state, _, _ = self.vae.encode(obs)
        h = self.rnn.init_hidden(self.z_size, self.device)
        done = False
        total_reward = 0

        for t in count():
            h = detach(h)
            if show:
                self.env.render()

            action = self.controller.select_action(state)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float, device=self.device)
            action = torch.tensor([action], dtype=torch.float, device=self.device)

            obs = self.resize_obs(obs)
            with torch.no_grad():
                next_state, _, _ = self.vae.encode(obs)

            #  _, next_h = self.rnn(next_state, h)
            self.memory.push(state, action, next_state, reward)
            state = next_state
            self.controller.optimize()

            if done:
                break;
        return total_reward
