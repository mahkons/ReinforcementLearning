import numpy as np
import gym
import math
import random
from collections import namedtuple
from itertools import count
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.normal import Normal
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from MDNRNN import MDNRNN, mdn_loss_fn, detach
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from vae.VAE import VAE

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def compare(x):
    with torch.no_grad():
        show(make_grid(x))
        plt.show()

z_size = 32
n_hidden = 256
n_gaussians = 5
BATCH_SIZE = 20

if __name__ == "__main__":
    device = torch.device("cpu")

    z = torch.load("generated/z.torch")
    mu = torch.load("generated/mu.torch")
    logvar = torch.load("generated/logvar.torch")
    z = z.view(BATCH_SIZE, -1, z.size(2)).to(device)

    vae_model = VAE(image_channels=3).to(device)
    vae_model.load_state_dict(torch.load('generated/vae.torch', map_location='cpu'))

    model = MDNRNN(z_size, n_hidden, n_gaussians)
    model.load_state_dict(torch.load('generated/mdnrnn.torch', map_location='cpu'))

    zero = np.random.randint(z.size(0))
    one = np.random.randint(z.size(1))
    x = z[zero:zero+1, one:one+1, :]
    y = z[zero:zero+1, one+1:one+2, :]

    hidden = model.init_hidden(1, device)
    (pi, mu, sigma), _ = model(x, hidden)

    y_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(n_gaussians)]

    compare(vae_model.decode(torch.cat([x, y] + y_preds)))
    
