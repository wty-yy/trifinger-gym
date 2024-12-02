import os
import random
import time
from dataclasses import dataclass
import cv2
import gym
import isaacgym  # noqa
from isaacgym import gymapi
#import isaacgymenvs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import sys
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad
# from leibnizgym.utils.rlg_train import create_rlgpu_env2
from leibnizgym.utils import rlg_train
from tqdm import trange
from pathlib import Path
import sys

SAC_path=Path(__file__).parent
sys.path.append(str(SAC_path))

model_name="sac_cube_noise_step310000_288.pt"
model_path=SAC_path/'models'/model_name

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        print("start_dim:",np.array(env.observation_space.shape).prod())
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        print("env.action_space.high",env.action_space.high)
        print("env.action_space.low",env.action_space.low)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        state_dict=torch.load(model_path,map_location=torch.device('cpu'))
        self.load_state_dict(state_dict=state_dict["actor"])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean