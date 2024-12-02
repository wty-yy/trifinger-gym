import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Envs:
    observation_space = (41,)
    action_space = (9,)

class Agent(nn.Module):
    def __init__(self, envs=Envs()):
        super().__init__()
        self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.Linear(np.array(envs.observation_space).prod(), 1),
            # nn.Linear(np.array(envs.observation_space).prod(), 256),
            # nn.Tanh(),
            # nn.Linear(256, 256),
            # # layer_init(nn.Linear(256, 256)),
            # nn.Tanh(),
            # # layer_init(nn.Linear(256, 1), std=1.0),
            # nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.action_space)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        print(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
agent = Agent()

def test():
    next_obs = np.loadtxt("/home/wq/Documents/leibnizgym/tests/next_obs.txt")
    next_obs = torch.tensor(next_obs).float()
    action, logprob, _, value = agent.get_action_and_value(next_obs)
    print(value)

if __name__ == '__main__':
    test()