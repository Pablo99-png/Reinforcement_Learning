import gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
#weight initializations
#uniform_distribution
#he_init(uniform and normal)
#Xavier/glorot_distribution(normal and uniform)

def glorot_uniform(n_inputs,n_outputs,multiplier=1.0):
    ''' Glorot uniform initialization '''
    glorot = multiplier*np.sqrt(6.0/(n_inputs+n_outputs))
    #Xavier_uniform
    return np.random.uniform(-glorot,glorot,size=(n_inputs,n_outputs))

def softmax(scores,temp=5.0): #normalized exponential function with temperature scaling to prevent overly confident prob. for high value scores.
    ''' transforms scores to probabilites '''
    exp = np.exp(np.array(scores)/temp)
    return exp/exp.sum()

class PolicyNet(nn.Module):
  def __init__(self,n_inputs, n_hidden, n_outputs):
    super(PolicyNetwork, self).__init__()
    self.fc1 = nn.Linear(n_inputs, n_hidden)
    self.fc2 = nn.Linear(n_hidden, n_outputs)
  
  def forward(self, x):
    x = self.tanh(self.fc1(x))
    x = self.tanh(self.fc2(x))
    return x

class ValueNet(nn.Module):
    def __init__(self, s_dim):
        super(ValueNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.main(state)[:, 0]


class AgentNN(nn.Module): #its the Agent network in the ES, and the PolicyNetwork in the PPO 
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)
    
    def loadFromTensors(self, W1, W2, b1, b2):
        self.fc1.weight = nn.Parameter(W1)
        self.fc2.weight = nn.Parameter(W2)
        self.fc1.bias = nn.Parameter(b1)
        self.fc2.bias = nn.Parameter(b2)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x



class Agent(object): 
    ''' A Neural Network '''
    #Activation= Tanh
    def __init__(self, n_inputs, n_hidden, n_outputs, mutate_rate=.05, init_multiplier=1.0):
        ''' Create agent's brain '''
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.mutate_rate = mutate_rate
        self.init_multiplier = init_multiplier
        self.network = {'Layer 1' : glorot_uniform(n_inputs, n_hidden,init_multiplier), #(25,512)
                        'Bias 1'  : np.zeros((1,n_hidden)),
                        'Layer 2' : glorot_uniform(n_hidden, n_outputs,init_multiplier), #(512,4)
                        'Bias 2'  : np.zeros((1,n_outputs))}
    
    def act(self, state):
        ''' Use the network to decide on an action '''
        #print(state)
        if type(state) == tuple: ### MIA
            state = state[0]  ### MIA
        if(state.shape[0] != 1):
            state = state.reshape(1,-1)
        net = self.network
        layer_one = np.tanh(np.matmul(state,net['Layer 1']) + net['Bias 1'])
        layer_two = np.tanh(np.matmul(layer_one, net['Layer 2']) + net['Bias 2'])
        return layer_two[0]

    def __add__(self, another):
        ''' overloads the + operator for breeding '''
        child = Agent(self.n_inputs, self.n_hidden, self.n_outputs, self.mutate_rate, self.init_multiplier)
        for key in child.network:
            n_inputs,n_outputs = child.network[key].shape
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[.5,.5])
            random = glorot_uniform(mask.shape[0],mask.shape[1]) #random weights initialized with glorot
            child.network[key] = np.where(mask==1,self.network[key],another.network[key]) #returns indices where mask =1
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[1-self.mutate_rate,self.mutate_rate]) #selects 0 with 1-mutationrate prob, 1 with mut. rate prob
            child.network[key] = np.where(mask==1,child.network[key]+random,child.network[key]) # updates child networks layers with weights=child.network[key]+random when mask 1
        return child

def run_trial(env,agent,verbose=False):
    print('fatto')
    ''' an agent performs 3 episodes of the env '''
    totals = []
    for _ in range(3):
        state = env.reset()
        if verbose: env.render()
        total = 0
        done = False
        while not done:
            #print(env.step(agent.act(state)))
            state, reward, _, done, _ = env.step(agent.act(state))
            if reward >= 100:
                env.render()
#             if verbose: env.render()
            total += reward
        totals.append(total)
    return sum(totals)/3.0

def next_generation(env,population,scores,temperature):
    ''' breeds a new generation of agents '''

    scores, population =  zip(*sorted(zip(scores,population),reverse=True)) #sort scores and population w.r.t. scores.
    #select the first 25% agents and mark as children
    children = list(population[:len(population)//4])
    #fill the remaining children with the best of parents.
    #A random sample is generated from population,with probabilities returned from softmax.
    #create 2 times the size of agents remaining after 25% children are removed.
    parents = list(np.random.choice(population,size=2*(len(population)-len(children)),p=softmax(scores,temperature))) #lista contenente la population semplata in base secondo il softmax dei loro scores
    #Breed between 2 Agent's from the above list and add it to the children list.
    children = children + [parents[i]+parents[i+1] for i in range(0,len(parents)-1,2)] #la somma è stata sovrascritta per accoppiare due modelli, vengono accoppiati i modelli vicini (1 e 2, 3 e 4,...)
    #run the children agents and return children agents and their scores.
    scores = [run_trial(env,agent) for agent in children]

    return children,scores #i childrens sono formati dal 25% dei modelli che hanno performato meglio e dal 75% dei modelli prodotti tramite accoppiamento dei modelli scelti in base al loro score


def update_plot(graph, new_data):
    graph.set_xdata(np.append(graph.get_xdata(), new_data[0]))
    graph.set_ydata(np.append(graph.get_ydata(), new_data[1]))
    plt.draw()

def trainES():#hidden = 512 input = 24 out = 4
    ''' main function '''
    graph = plt.plot([],[])
    # Setup environment
    env = gym.make('BipedalWalker-v3')
    np.random.seed(0)
    genlist=[]
    rewardlist=[]
    # network params
    n_inputs = env.observation_space.shape[0] # 24 observations
    n_actions = env.action_space.shape[0] # 4 actions
    n_hidden = 512
    multiplier = 5

    # Population params
    pop_size = 50
    mutate_rate = .1
    softmax_temp = 5.0

    # Training
    n_generations = 80
    #create agents(as per population size)
    startTime = time.time()
    population = [Agent(n_inputs,n_hidden,n_actions,mutate_rate,multiplier) for i in range(pop_size)]
    #run all agents in the population
    scores = [run_trial(env,agent) for agent in population]
    #choose the best agent from the above trial and store it as best agent.
    best = [deepcopy(population[np.argmax(scores)])]
    #create new generation and repeat for n generations
    for generation in range(n_generations):
        #create next generation fromcurrent poulation and scores.
        json.dump({"time":time.time()-startTime,"meanReward":sum(scores)/len(scores)}, open('rewardsES.json', 'a'))
        population,scores = next_generation(env,population, scores,softmax_temp)
        best.append(deepcopy(population[np.argmax(scores)]))
        print( "Generation:",generation,"Score:",np.max(scores))
        genlist+=[generation]
        rewardlist+=[np.max(scores)]
        xpoints = np.array(genlist)
        ypoints = np.array(rewardlist)
        plt.plot(xpoints, ypoints)
        plt.title("Genetic Algorithm")
        plt.xlabel("Generation")
        plt.ylabel("Reward")
        if generation>= 70:
            plt.show()
        if generation%5==0:
            BESTMODEL = population[np.argmax(scores)]
            torch.save(AgentNN().loadFromTensors(W1=BESTMODEL.network['Layer 1'],W2=BESTMODEL.network['Layer 2'],b1=BESTMODEL.network['Bias 1'],b2=BESTMODEL.network['Bias 2']).state_dict(), f'bestAgent{generation}.pt')

        #         data= (generation, np.max(scores))
#         update_plot(graph,data)

    # Record every agent
    env = gym.wrappers.Monitor(env,'/monitor_output',force=True,video_callable=lambda episode_id: episode_id%3==0) #'/tmp/walker'
    for agent in best:
        run_trial(env,agent)
    env.close()

if __name__ == '__main__':
    trainES()