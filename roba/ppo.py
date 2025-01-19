import gym
import torch
from torch import nn

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

#Gaussian distribution with given mean & std.
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, x):
        return super().log_prob(x).sum(-1)
    
    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(inp_dim, out_dim)
        self.b_logstd = AddBias(torch.zeros(out_dim))
    
    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.b_logstd(torch.zeros_like(mean))
        return FixedNormal(mean, logstd.exp())

class PolicyNet(nn.Module):
    #Constructor
    def __init__(self, s_dim, a_dim):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.dist = DiagGaussian(128, a_dim)
    
    #Forward pass
    def forward(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        return action, dist.log_probs(action)
    
    #Choose an action (stochastically or deterministically)
    def choose_action(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            return dist.mode()

        return dist.sample()
    
    #Evaluate a state-action pair (output log-prob. & entropy)
    def evaluate(self, state, action):
        feature = self.main(state)
        dist = self.dist(feature)
        return dist.log_probs(action), dist.entropy()


env  = gym.make('BipedalWalker-v3', render_mode="human")
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

vNet = PolicyNet(24,4)
vNet.load_state_dict(torch.load('savesPPO/saveAAAAAAA/modelsPN4000.pt')['PolicyNet']) #2000

state = env.reset()
done = False
while not done:
  with torch.no_grad():
    #print(f'state = --{state}--')
    if type(state) == tuple:
        state = state[0]
    action = vNet(torch.from_numpy(state).unsqueeze(0))[0].tolist()[0]
    state, reward, _, done,_ = env.step(action)
    env.render()
