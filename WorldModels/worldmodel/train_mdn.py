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

values = np.sin(np.arange(100000) / 100)

epochs = 500
seqlen = 16
BATCH_SIZE = 200

z_size = 32
n_hidden = 256
n_gaussians = 5
