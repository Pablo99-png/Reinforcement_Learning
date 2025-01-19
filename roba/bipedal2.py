"""import gym

env = gym.make("BipedalWalker-v3")
env.reset()
while True: 
    action = env.action_space.sample()
    print(action)
    1/0
    print(env.step(action))
    env.render()
    """
"""
import torch
import torch.nn as nn


W1 = torch.randn(10,20)
b1 = torch.randn(20)
W2 = torch.randn(20,10)
b2 = torch.randn(10)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(10,20)
        self.fc2 = nn.Linear(20,10)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

MyNN = NN()
with torch.no_grad():
    MyNN.fc1.weight = nn.Parameter(W1)
    MyNN.fc1.bias = nn.Parameter(b1)
    MyNN.fc2.weight = nn.Parameter(W2)
    MyNN.fc2.bias = nn.Parameter(b2)

    """
import json
import matplotlib.pyplot as plt 

f = open('/home/pablo/Downloads/rewards.json','r')
x = []
min = {'r':-10000,'n':-1}
i = 0
for line in f:
    if '}{' in line:
        line = line.split('}{')
        r = json.loads(line[0] + '}')['meanReward']
        x.append(r)
    else:  
        r = json.loads(line)['meanReward']
        x.append(r)
    if x[-1] > min['r']:
        min['r'] = r
        min['n'] = i*200
    i+=1

print(f'toppe = {min["n"]}' )
plt.plot([200*i for i in range(25)],x)
plt.show()