import gym
import numpy as np
from torch import nn
from torch import functional as F
import torch

env_name = "BipedalWalker-v3"
env = gym.make(env_name)#, render_mode="human")

class AgentNN(nn.Module): #its the Agent network in the ES, and the PolicyNetwork in the PPO
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(24, 512)
        self.fc2 = nn.Linear(512, 4)
        self.tanh = nn.Tanh()

    def loadFromTensors(self, W1, W2, b1, b2):
        self.fc1.weight = nn.Parameter(W1.T)
        self.fc2.weight = nn.Parameter(W2.T)
        self.fc1.bias = nn.Parameter(b1)
        self.fc2.bias = nn.Parameter(b2)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x


params = torch.load('/home/pablo/Downloads/bestAgent43.pt')
params['fc1.weight'] = params['fc1.weight'].T
params['fc2.weight'] = params['fc2.weight'].T

params['fc1.bias'] = params['fc1.bias'].squeeze(0)
params['fc2.bias'] = params['fc2.bias'].squeeze(0)

rete = AgentNN(512, 24, 4)
rete.load_state_dict(params)
allRewards = []
for _ in range(100):
    done = False
    state = env.reset()
    rewardRun = 0
    while not done:
        with torch.no_grad():
            if type(state) == tuple:
                state = state[0]
            action = rete(torch.from_numpy(state))
            #print(action)
            state, reward, done, info,_ = env.step(action)
            env.render()    
            rewardRun+=reward
    print(rewardRun)
    allRewards.append(rewardRun)

    
print('rewardMedio:',sum(allRewards)/len(allRewards))

import matplotlib.pyplot as plt

plt.hist(allRewards, bins=10)
plt.xlabel('Rewards')
plt.ylabel('Frequency')
plt.title('Histogram of Rewards')
plt.show()