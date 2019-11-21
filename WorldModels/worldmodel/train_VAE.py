import plotly
import numpy as np
import math
import gym
import random
import Box2D
import matplotlib.pyplot as plt
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
dataset = datasets.ImageFolder(root='./rollouts', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
len(dataset.imgs), len(dataloader)


img, _ = next(iter(dataloader))
grid = torchvision.utils.make_grid(img)
show(grid)

model = VAE(image_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def compare(x=None):
    with torch.no_grad():
        if x is None:
            x = dataset[random.randint(1, 100)][0].unsqueeze(0)
        recon_x, _, _ = model(x)
        show(make_grid(torch.cat([x, recon_x])))
        plt.show()


epochs = 50

if __name__ == "__main__":
    for epoch in range(epochs):
        for idx, (images, _) in enumerate(dataloader):
            recon_images, mu, logvar = model(images)

            loss = VAE_loss(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.item()/BATCH_SIZE)
            print(to_print)

        compare()
        torch.save(model.state_dict(), 'vae.torch')
