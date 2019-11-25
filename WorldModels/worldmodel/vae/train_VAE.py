import plotly
import numpy as np
import math
import gym
import random
import Box2D
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.core.display import Image, display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import make_grid

from VAE import VAE, VAE_loss

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

BATCH_SIZE = 32
dataset = datasets.ImageFolder(root='vae/rollouts', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
len(dataset.imgs), len(dataloader)

def compare(model, x=None):
    with torch.no_grad():
        if x is None:
            x = dataset[random.randint(1, 100)][0].unsqueeze(0)
        recon_x, _, _ = model(x)
        show(make_grid(torch.cat([x, recon_x])))
        plt.show()

img, _ = next(iter(dataloader))
grid = torchvision.utils.make_grid(img)
show(grid)
plot_data=[]

def train(epochs=1, restart=True):
    model = VAE(image_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if not restart:
        model.load_state_dict(torch.load("generated/vae.torch", map_location='cpu'))

    for epoch in range(epochs):
        for idx, (images, _) in enumerate(dataloader):
            recon_images, mu, logvar = model(images)

            loss = VAE_loss(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            plot_data.append(loss.item())
            to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/BATCH_SIZE)
            print(to_print)

        compare(model)
    torch.save(model.state_dict(), 'generated/vae.torch')

if __name__ == "__main__":
    train(3, True)
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(plot_data)))
    plot.show()
