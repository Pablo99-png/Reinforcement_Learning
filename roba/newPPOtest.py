"""import numpy as np
import gym
from tensorboardX import SummaryWriter

import datetime
from collections import namedtuple
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import matplotlib.pyplot as plt
class A2C_policy(nn.Module):
    '''
    Policy neural network
    '''
    def __init__(self, input_shape, n_actions):
        super(A2C_policy, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU())

        self.mean_l = nn.Linear(32, 4)
        self.mean_l.weight.data.mul_(0.1)

        self.var_l = nn.Linear(32, 4)
        self.var_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))


agent = A2C_policy(24,4)
agent.load_state_dict(torch.load('/home/pablo/Downloads/PPO_BipedalWalker-v3_15_17.44.56_0.0004_0.001_2049_64.pth.tar',map_location=torch.device('cpu'))['agent_policy'] )


env = gym.make('BipedalWalker-v3')#, render_mode="human")

rewards = []
for i in range(100):
    state = env.reset()
    done = False
    rewardRun = 0
    while not done:
        with torch.no_grad():
            if type(state) == tuple:
                state = state[0]
            action = agent(torch.from_numpy(state).unsqueeze(0))[0].tolist()
            state, reward, done, info,_ = env.step(action)
            env.render()
            rewardRun+=reward
    print(rewardRun)
    rewards.append(rewardRun)

print(f'Mean reward: {sum(rewards)/len(rewards)}')

plt.hist(rewards, bins=10)
plt.show()
"""
import json
import matplotlib.pyplot as plt

a = """
{"step": 1800, "time": 1.8462412357330322, "meanReward": 97}{"step": 400, "time": 0.3964574337005615, "meanReward": 73}{"step": 1200, "time": 1.1967196464538574, "meanReward": 73}{"step": 1800, "time": 1.792872667312622, "meanReward": 90}{"step": 800, "time": 0.7566819190979004, "meanReward": 88}{"step": 1200, "time": 1.2003552913665771, "meanReward": 43}{"step": 400, "time": 0.409942626953125, "meanReward": -27}{"step": 2000, "time": 2.02824068069458, "meanReward": -32}{"step": 800, "time": 0.7756826877593994, "meanReward": 17}{"step": 1600, "time": 1.5957787036895752, "meanReward": 111}{"step": 600, "time": 0.6270430088043213, "meanReward": 145}{"step": 1600, "time": 1.5980019569396973, "meanReward": 25}{"step": 1200, "time": 1.2363224029541016, "meanReward": 67}{"step": 1000, "time": 1.0090105533599854, "meanReward": 74}{"step": 200, "time": 0.19597268104553223, "meanReward": -58}{"step": 2000, "time": 2.0609652996063232, "meanReward": -101}{"step": 400, "time": 0.4187319278717041, "meanReward": -92}{"step": 1800, "time": 1.887993335723877, "meanReward": -108}{"step": 1000, "time": 1.1564838886260986, "meanReward": -113}{"step": 800, "time": 0.8794410228729248, "meanReward": -114}{"step": 800, "time": 0.8393709659576416, "meanReward": -115}{"step": 1800, "time": 1.9152045249938965, "meanReward": -112}{"step": 1600, "time": 1.713921308517456, "meanReward": -111}{"step": 400, "time": 0.3960990905761719, "meanReward": -113}{"step": 2000, "time": 1.9771201610565186, "meanReward": -113}{"step": 200, "time": 0.2163102626800537, "meanReward": -108}{"step": 600, "time": 0.6634397506713867, "meanReward": -114}{"step": 1200, "time": 1.3303461074829102, "meanReward": -114}{"step": 1600, "time": 2.088567018508911, "meanReward": -109}{"step": 1000, "time": 1.0801072120666504, "meanReward": -106}{"step": 800, "time": 1.0181663036346436, "meanReward": -105}{"step": 0, "time": 0.0046291351318359375, "meanReward": -116}{"step": 1000, "time": 1.0903029441833496, "meanReward": -119}{"step": 1400, "time": 1.5130815505981445, "meanReward": -119}{"step": 2000, "time": 2.1439802646636963, "meanReward": -119}{"step": 1200, "time": 1.5676374435424805, "meanReward": -121}{"step": 1200, "time": 1.2944879531860352, "meanReward": -121}{"step": 2000, "time": 2.152752637863159, "meanReward": -119}{"step": 0, "time": 0.004270315170288086, "meanReward": -119}{"step": 1400, "time": 1.4771065711975098, "meanReward": -120}
"""

a = a.split('}{')

k = [json.loads(a[0] + '}')] 

k += [json.loads('{' + x + '}') for x in a[1:-1]]

k += [json.loads('{'+a[-1])] 

x = [i['meanReward'] for i in k]
x = list(reversed(x))
x += [150,138,115,160,171,180,164,120,270,280,298,270,248,253]
plt.plot(x)
plt.show()

