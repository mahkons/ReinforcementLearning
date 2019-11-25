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

from MDNRNN import MDNRNN, mdn_loss_fn, detach

device = torch.device("cpu")

epochs = 500
seqlen = 16
BATCH_SIZE = 20

z_size = 32
n_hidden = 256
n_gaussians = 5

z = torch.load("generated/z.torch")
mu = torch.load("generated/mu.torch")
logvar = torch.load("generated/logvar.torch")

plot_data = []
z = z.view(BATCH_SIZE, -1, z.size(2)).to(device)

def train(epochs=10, restart=True):
    model = MDNRNN(z_size, n_hidden, n_gaussians)
    if not restart:
        model.load_state_dict(torch.load("generated/mdnrnn.torch", map_location='cpu'))
    criterion = mdn_loss_fn
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        hidden = model.init_hidden(BATCH_SIZE, device)
        for i in range(0, z.size(1) - seqlen, seqlen):
            inputs = z[:, i:i+seqlen, :]
            targets = z[:, (i+1):(i+1)+seqlen, :]
            
            hidden = detach(hidden)
            (pi, mu, sigma), hidden = model(inputs, hidden)
            loss = criterion(targets.unsqueeze(2), pi, mu, sigma)
            
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            plot_data.append(loss.item())
            
        if epoch % 5 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
                .format(epoch+1, epochs, loss.item()))

    torch.save(model.state_dict(), "generated/mdnrnn.torch")

train(100, restart=False)

plot = go.Figure()
plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(plot_data)))
plot.show()
