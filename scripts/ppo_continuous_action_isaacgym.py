# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
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
from tqdm import trange
from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad
sys.path.append('/home/wq/Documents/leibnizgym/leibnizgym/utils')
from rlg_train import create_rlgpu_env2


import csv
import pandas as pd
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 7
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    dataset_save:  bool=True
    """whether save[obs,action,reward]as dataset"""


    # Algorithm specific arguments
    env_id: str = "Ant"
    """the id of the environment"""
    total_timesteps: int = 300000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0026
    """the learning rate of the optimizer"""
    num_envs: int = 4096
    """the number of parallel game environments"""
    num_steps: int = 400
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True #False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 2
    """coefficient of the value function"""
    max_grad_norm: float = 1
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    reward_scaler: float = 1
    """the scale factor applied to the reward during training"""
    record_video_step_frequency: int = 1464
    """the frequency at which to record the videos"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= ~dones
        self.episode_lengths *= ~dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class   Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            # nn.Linear(np.array(envs.observation_space.shape).prod(), 256),
            nn.Tanh(),
            # nn.Linear(256, 256),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
            # nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]

def test(hydra_cfg):
    
    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    rlg_cfg = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    envs=create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)
    record=True
    if record:
        import csv
        from pathlib import Path
        path=Path(__file__).parents[1]
        path=path/'experient'
        path=os.path.join(path,'ppo_cube_noise.csv')
        
        with open(path,"w")as csvfile:
            writer=csv.writer(csvfile)
            b=['target','object','success','steps']
            writer.writerow(b)
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if 'checkpoint' in cli_args and cli_args['checkpoint'] is not None and cli_args['checkpoint'] !='':
        filename=cli_args['checkpoint']
        print(f"载入模型：{filename}")
        checkpoint=torch.load(filename,map_location=torch.device('cpu'))
        agent.load_state_dict(checkpoint)
    print("start ppo test")
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
        object=obs_trans[18:21].detach().squeeze().cpu().numpy()
        target=obs_trans[25:28].detach().squeeze().cpu().numpy()
        # print(target,object)

        success=0
        while (success==0) &(step<100):
            # print(obs)

            action=agent.actor_mean(obs)
            action=torch.clip(action,-1,1)
            nex_obs, rewards, next_done, info = envs.step(action)
            nex_obs[:,9:18]=0
            success=next_done.detach().squeeze().cpu().numpy()
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
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs=create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)
    agent = Agent(envs).to(device)
    #optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if 'checkpoint' in cli_args and cli_args['checkpoint'] is not None and cli_args['checkpoint'] !='':
        filename=cli_args['checkpoint']
        print(f"载入模型：{filename}")
        checkpoint=torch.load(filename,map_location=torch.device('cpu'))
        agent.load_state_dict(checkpoint)

    if args.dataset_save:

        with open("/home/wq/Documents/dataset_v7.csv","w")as csvfile:
            writer=csv.writer(csvfile)
            b=[str(x) for x in range(93)]
            writer.writerow(b)
            

    obs=envs.reset()
    obs[:,9:18]=0
    nex_obs=obs.clone()
    image_idx=0
    ###success rate cal

    success_dic=dict()
    idx=1
    for i in range(750*1000):
        
        obs=nex_obs
        action=agent.actor_mean(obs)
        action=torch.clip(action,-1,1)

        obs_trans=unscale_transform(
                        obs[0],
                        lower=envs.obs_scale.low,
                        upper=envs.obs_scale.high
                    )

        #print(obs_trans)


        nex_obs, rewards, next_done, info = envs.step(action)
        nex_obs[:,9:18]=0
        
        # if np.min(nex_obs.detach().numpy())<-1.5:
        #     image_idx+=1
        #     print("index",np.argwhere(nex_obs<-1.5))
        #     obs_trans=unscale_transform(
        #                 nex_obs[0],
        #                 lower=envs.obs_scale.low,
        #                 upper=envs.obs_scale.high
        #             )
        #     print("obs_trans",obs_trans)
        #     color_image = gym.get_camera_image(envs.sim, envs.camera_handle, gymapi.IMAGE_COLOR)
        #     cv2.imwrite(f"trifinger_{image_idx}_debug",color_image)\
        
  

        if args.dataset_save:
            
            data=torch.cat([obs.squeeze(),action.squeeze(),rewards,next_done,nex_obs.squeeze()]).detach().numpy()
            with open("/home/wq/Documents/dataset_v7.csv","a")as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(data)
    

        # print(action)
        # print(nex_obs)
        # b=0
        # with torch.no_grad():
        #     while True:
        #         if gym_cfg["normalize_action"]:
        #             # TODO: Default action should correspond to normalized value of 0.
        #             action_transformed = unscale_transform(
        #                 action,
        #                 lower=envs.action_scale.low,
        #                 upper=envs.action_scale.high
        #             )
        #         else:
        #             action_transformed=action
        #         action_transformed=np.clip(action_transformed,envs.action_scale.low,envs.action_scale.high)
        #         if torch.all(torch.abs(action_transformed-envs.dof_position)<0.1):
        #             print(b)
        #             b=0
        #             break
        #         elif b>200:
        #             #rewards[step]-=0.3
        #             print(f"bad_action{action_transformed}\n pos:{envs.dof_position}")
                    
        #             break
        #         else:
        #             b+=1
        #             nex_obs, reward, next_done, info = envs.step(action)
        #             nex_obs[:,9:18]=0
        
        #print(action)
        #print(rewards)


def train(hydra_cfg):
    from omegaconf import OmegaConf
    gym_cfg = OmegaConf.to_container(hydra_cfg.gym)
    rlg_cfg = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    args = tyro.cli(Args)
    args.num_envs=cli_args.num_envs
    
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #env = TrifingerEnv(config=gym_cfg, device='cpu', verbose=True, visualize=True)
    envs=create_rlgpu_env2(task_cfg=gym_cfg,cli_args=cli_args)
    '''
    envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.env_id,
        num_envs=args.num_envs,
        sim_device="cuda:0" if torch.cuda.is_available() and args.cuda else "cpu",
        rl_device="cuda:0" if torch.cuda.is_available() and args.cuda else "cpu",
        graphics_device_id=0 if torch.cuda.is_available() and args.cuda else -1,
        headless=False if torch.cuda.is_available() and args.cuda else True,
        multi_gpu=False,
        virtual_screen_capture=args.capture_video,
        force_render=False,
    )
    '''
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )
    #envs = ExtractObsWrapper(envs)
    #envs = RecordEpisodeStatisticsTorch(envs, device)
    #envs.single_action_space = envs.action_space
    #envs.single_observation_space = envs.observation_space
    single_action_space = envs.action_space
    single_observation_space = envs.observation_space
    print(single_action_space,single_observation_space)
    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if cli_args.checkpoint!="":
        filename=cli_args['checkpoint']
        print(f"载入模型：{filename}")
        checkpoint=torch.load(filename,map_location=torch.device('cpu'))
        agent.load_state_dict(checkpoint)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs[:,9:18]=0
    next_done = torch.zeros(args.num_envs, dtype=torch.bool).to(device)
    print(f'num_iterations:{args.num_iterations}\t ')
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, info = envs.step(action)
            next_obs[:,9:18]=0
  
            
            """  
            if 0 <= step <= 2:
                for idx, d in enumerate(next_done):
                    if d:
                        episodic_return = info["r"][idx].item()
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                        if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                            writer.add_scalar(
                                "charts/consecutive_successes", info["consecutive_successes"].item(), global_step
                            )
                        break
            """

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = ~next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = ~dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                # print(advantages[-5:], delta, lastgaelam, rewards[-5:], values[-5:])
                # if args.num_steps - t > 3: exit()
            returns = advantages + values
        # print(obs)

        # print("reward", rewards[-10:])
        # print("mean reward", rewards.mean(0))
        # print("cumulate reward", rewards.sum(0))
        # print("values", values[-10:])
        # print("advantage", advantages[-10:])
        # print("returns", returns[-10:])

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    print(newvalue.mean())
                    print(b_returns[mb_inds].mean())
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                print(f"loss:{loss},pg_loss:{pg_loss},v_loss:{v_loss}")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("reward",torch.mean(rewards).item(),global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("-"*20)
        print(f"iteration:{iteration}")
        #print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if iteration % 30 == 0:
            torch.save(agent.state_dict(), './PPO_continuous_itera{}.pth'.format(iteration))

    # envs.close()
    writer.close()
