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
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import sys
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad
# from leibnizgym.utils.rlg_train import create_rlgpu_env2
sys.path.append('/home/wq/Documents/leibnizgym/leibnizgym/utils')
from rlg_train import create_rlgpu_env2

from leibnizgym.CQL.predict import TanhGaussianPolicy



import csv
import pandas as pd

def play(hydra_cfg):

    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    rlg_cfg = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    envs=create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)

    state_dim=41
    action_dim=9
    max_action=1.0
    policy_log_std_multiplier: float = 1.0
    orthogonal_init: bool = True  # Orthogonal initialization
    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=policy_log_std_multiplier,
        orthogonal_init=orthogonal_init,
    )
    
    #optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

         

    obs=envs.reset()
    obs[:,9:18]=0
    nex_obs=obs.clone()
    image_idx=0
    rewards=torch.tensor([0])
    for _ in range(750*1000):
        
        obs=nex_obs
        #print("obs min max", obs.min(), obs.max())
        #print(rewards[0].detach().numpy())
        action=actor.act(obs.squeeze().detach().numpy())
        print(action)
        action = action + np.random.randn(9).astype(np.int32) * 0.3
        #action=action*100
        action=torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        action=torch.clip(action,-1,1)


        obs_trans=unscale_transform(
                        obs[0],
                        lower=envs.obs_scale.low,
                        upper=envs.obs_scale.high
                    )

        #print(obs_trans)

        nex_obs, rewards, next_done, info = envs.step(action)
        nex_obs[:,9:18]=0
        #print("-"*20)
        #print(action[0],rewards[0])
        #print(nex_obs.shape)
        #print(rewards.shape)

        

            

    