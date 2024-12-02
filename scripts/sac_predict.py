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
from leibnizgym.SAC.predict import Actor
from tqdm import trange


def test(hydra_cfg):
    
    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    rlg_cfg = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    envs=rlg_train.create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)
    record=False
    if record:
        import csv
        from pathlib import Path
        path=Path(__file__).parents[1]
        path=path/'experient'
        path=os.path.join(path,'sac_check.csv')
        
        with open(path,"w")as csvfile:
            writer=csv.writer(csvfile)
            b=['target','object','success','steps']
            writer.writerow(b)
    actor = Actor(envs).to(device)

    for _ in trange(5000):
        obs=envs.reset()
        obs[:,9:18]=0
        # print(obs)
        get_pos=obs.clone()
        obs_trans=unscale_transform(
                        get_pos[0],
                        lower=envs.obs_scale.low,
                        upper=envs.obs_scale.high
                    )
        step=0
        object=obs_trans[18:21].detach().squeeze().numpy()
        target=obs_trans[25:28].detach().squeeze().numpy()
        # print(target,object)

        success=0
        while (success==0) &(step<100):
            # print(obs)
            print("obs:",obs)
            action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            action=torch.clip(action,-1,1)
            print("action:",action)
            nex_obs, rewards, next_done, info = envs.step(action)
            nex_obs[:,9:18]=0
            success=next_done
            obs=nex_obs.clone()
            step+=1
        if record:
            
            data=np.append(target,object)
            data=np.append(data,success)
            data=np.append(data,step)
            # print(data)
            with open(path,"a")as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(data)
        

        


def play(hydra_cfg):
    
    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    rlg_cfg = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    envs=rlg_train.create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)

    state_dim=41
    action_dim=9
    max_action=1.0
    hidden_dim: int = 256
    actor = Actor(envs).to(device)
    
    
    #optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

         

    obs=envs.reset()
    # obs[:,9:18]=0
    nex_obs=obs.clone()
    image_idx=0
    rewards=torch.tensor([0])
    print("start sac predict")
    for _ in range(750*1000):
        
        obs=nex_obs

        action, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        action = action.detach()
        # print(action)

        
        action=torch.clip(action,-1,1)


        obs_trans=unscale_transform(
                        obs[0],
                        lower=envs.obs_scale.low,
                        upper=envs.obs_scale.high
                    )

        #print(obs_trans)

        nex_obs, rewards, next_done, info = envs.step(action)
        #nex_obs[:,9:18]=0


