import numpy as np
import gym
import math
import random
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
import argparse

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



state_sz = 32
action_sz = 3
hidden_sz = 256
n_gaussians = 5

epochs = 1
render_env = False

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, required=False)
    parser.add_argument('--show', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    return parser


if __name__ == "__main__":

    env = gym.make('CarRacing-v0')
    #  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    args = create_parser().parse_args()

    print("Device: {}".format(device))
    print("Show: {}. Restart {}".format(args.show, args.restart))

    epochs, render_env = args.epochs, args.show

    vae = VAE(image_channels=3)
    vae.load_state_dict(torch.load('generated/vae.torch', map_location='cpu'))
    vae.to(device)

    mdnrnn = MDNRNN(state_sz, hidden_sz, n_gaussians)
    mdnrnn.load_state_dict(torch.load('generated/mdnrnn.torch', map_location='cpu'))
    mdnrnn.to(device)

    controller = ControllerAC(env, state_sz, action_sz, hidden_sz, device=device)
    if not args.restart:
        controller.load_model("generated/dqn.torch")
    agent = Agent(env, mdnrnn, vae, controller, device=device)

    plot_data = list()
    for episode in range(epochs):
        reward = agent.rollout(show=render_env)
        print("Episode {}/{}. Reward: {}".format(episode, epochs, reward))
        plot_data.append(reward)

    controller.save_model("generated/dqn.torch")
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(epochs), y=np.array(plot_data)))
    plot.show()
