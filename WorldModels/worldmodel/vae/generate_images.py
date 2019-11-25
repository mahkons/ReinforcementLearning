# -*- coding: utf-8 -*-
from CarRacing import CarRacing

import imageio
import numpy as np
from random import choice, random, randint

env = CarRacing()

episodes = 100
steps = 150

def get_action(steps):
    action = env.action_space.sample()
    if steps < 70:
        action[0] = 0
    action[1] = 1
    action[2] = 0
    return action

if __name__ == "__main__":

    for eps in range(episodes):
        obs = env.reset()
        env.render()
        r = 0
        for t in range(steps):
            action = get_action(t)
            obs, reward, done, _ = env.step(action)
            env.render()
            r += reward
            if t%1 == 0:
                i = ('000' + str(t))[-3:]
                imageio.imwrite(f'./rollouts/SimplePolicy/car_{eps}_{i}.jpg', obs)
        print("Episode [{}/{}]: CummReward {:.2f}".format(eps+1, episodes, r))

