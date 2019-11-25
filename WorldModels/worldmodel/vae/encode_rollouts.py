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

BATCH_SIZE = 1
dataset = datasets.ImageFolder(root='vae/rollouts', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
len(dataset.imgs), len(dataloader)


vae = VAE(image_channels=3)
vae.load_state_dict(torch.load('generated/vae.torch', map_location='cpu'))


z_l, mu_l, logvar_l = [], [], []
for (images, _) in dataloader:
    z, mu, logvar = vae.encode(images)
    z_l.append(z)
    mu_l.append(mu)
    logvar_l.append(logvar)

print(z_l[0].shape, z_l[1].shape)
print(torch.stack(z_l).shape)

z_l = torch.stack(z_l)
mu_l = torch.stack(mu_l)
logvar_l = torch.stack(logvar_l)

torch.save(z_l, "generated/z.torch")
torch.save(mu_l, "generated/mu.torch")
torch.save(logvar_l, "generated/logvar.torch")

with torch.no_grad():
    x = np.random.randint(z.size(0))
    show(make_grid(vae.decode(z_l[x:x+16])))
    plt.show()


