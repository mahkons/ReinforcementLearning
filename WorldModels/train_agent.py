import numpy as np
import gym
import math
import random
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_

from agent.Agent import Agent
from worldmodel.vae.VAE import VAE
from worldmodel.mdnrnn.MDNRNN import MDNRNN
from agent.Controller import ControllerAC
from agent.ControllerDQN import ControllerDQN

import plotly as plt
import plotly.graph_objects as go

z_size = 32
n_hidden = 256
n_gaussians = 5

epochs = 50

if __name__ == "__main__":

    env = gym.make('CarRacing-v0')
    #  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Device: {}".format(device))

    vae = VAE(image_channels=3)
    vae.load_state_dict(torch.load('worldmodel/generated/vae.torch', map_location='cpu'))
    vae.to(device)

    mdnrnn = MDNRNN(z_size, n_hidden, n_gaussians)
    mdnrnn.load_state_dict(torch.load('worldmodel/generated/mdnrnn.torch', map_location='cpu'))
    mdnrnn.to(device)

    controller = ControllerDQN(env, z_size, 3, device=device)
    agent = Agent(env, mdnrnn, vae, controller, device=device)

    plot_data = list()
    for episode in range(epochs):
        reward = agent.rollout(show=True)
        print("Episode {}. Reward: {}".format(episode, reward))
        plot_data.append(reward)

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(epochs), y=np.array(plot_data)))
    plot.show()
