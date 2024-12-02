"""
@brief      Demo script for checking tri-finger environment.
"""

# leibnizgym
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad
# python
import torch
import numpy as np

import sys
def data_back(data,data_low,data_high):
        m=(data_low+data_high)/2
        d=(data_high-data_low)/2
        return data*d+m
def run(env_cfg):
    env = TrifingerEnv(config=env_cfg, device='cpu', verbose=True, visualize=True)
    _ = env.reset()
    print_info("Trifinger environment creation successful.")
    if 0:
        print("dof_position:",env._dof_position)
        print("action_shape:",env.get_action_shape())
        action=torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
        action[:,0]=1
        env.render()
        _, _, _, _ = env.step(action)
        env.render()

        print("action:",action)
        print("dof_position:",env._dof_position)

        print("dof_vel:",env._dof_velocity)
    # sample run
    from isaacgym import gymtorch

    a=10000
    start=env._dof_position.clone()
    print("dof_position:",env._dof_position)
    #print(data_back(env._dof_position,env._robot_limits["joint_position"].low,env._robot_limits["joint_position"].high))
    print("action_shape:",env.get_action_shape())
    poses=torch.tensor([0.5,  0.90, -1.0,0.5675,  0.6560, -1.4200,0.5675,  0.6560, -1.6200],dtype=torch.float32)
    action=torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
    b=0
    while a>0:
        _, _, _, _ = env.step(action)

        if env_cfg["normalize_action"]:
            # TODO: Default action should correspond to normalized value of 0.
            action_transformed = unscale_transform(
                action,
                lower=env._action_scale.low,
                upper=env._action_scale.high
            )
        if torch.all(torch.abs(action_transformed-env._dof_position)<0.1):
            print(b)
            b=0
            action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        elif b>2:
            print(f"bad_pos{action_transformed}\npos:{env._dof_position}")
            b=0
            action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        else:
            b+=1
        
        #action=torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
        #action[:,0]=1
        # step through physics
        #action=torch.tensor([ 0.2593,  0.0000,  0.0000,  0.4139,  0.0000,  0.0000, -0.3300,  0.0000,0.0000],dtype=torch.float32).unsqueeze(0)
        
        
        """
        env._gym.refresh_dof_state_tensor(env._sim)
        _dof_states=env._gym.acquire_dof_state_tensor(env._sim)
        dof_states=gymtorch.wrap_tensor(_dof_states)
        print(dof_states)
        print(dof_states.shape)
        dof_states[:,0]=poses
        env._gym.set_dof_state_tensor(env._sim,gymtorch.unwrap_tensor(dof_states))
        env._gym.simulate(env._sim)
        """




        # render environment
        env.render()
        a=a-1
        #print(a)
    print("dof_position:",env._dof_position)
    deta=env._dof_position-start
    print("deta:",deta)

if __name__ == '__main__':
    # configure the environment
    env_config = {
        'num_instances': 1,
        'aggregrate_mode': True,
        'control_decimation': 1,
        'command_mode': 'position',
        'sim': {
            "use_gpu_pipeline": False,
            "physx": {
                "use_gpu": False,
            },
        "reset_distribution": {
                # Defines how to reset the robot joint state
                "robot_initial_state": {
                    "type": "default",
                },
                # Defines how to reset the robot joint state
                "object_initial_state": {
                    "type": "default",
                }
            }
        }
        
    }
    # create environment
    env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)
    _ = env.reset()
    print_info("Trifinger environment creation successful.")
    if 0:
        print("dof_position:",env._dof_position)
        print("action_shape:",env.get_action_shape())
        action=torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
        action[:,0]=1
        env.render()
        _, _, _, _ = env.step(action)
        env.render()

        print("action:",action)
        print("dof_position:",env._dof_position)

        print("dof_vel:",env._dof_velocity)
    # sample run
    a=5000
    start=env._dof_position.clone()
    print("dof_position:",env._dof_position)
    #print(data_back(env._dof_position,env._robot_limits["joint_position"].low,env._robot_limits["joint_position"].high))
    print("action_shape:",env.get_action_shape())
    from isaacgym import gymtorch
    while 1:
        # zero action agent
        #action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        #action=torch.zeros(env.get_action_shape(), dtype=torch.float, device=env.device)
        #action[:,6]=1
        # step through physics
        #_, _, _, _ = env.step(action)
        # render environment

        # applied_torque=torch.ones(env.get_action_shape(),dtype=torch.float32)*-1

        # env._gym.set_dof_position_target_tensor(env._sim,gymtorch.unwrap_tensor(applied_torque))

        # env._gym.simulate(env._sim)
        # env._gym.refresh_dof_state_tensor(env._sim)
        # print(env._dof_position)
        
        
        env.render()
        a=a-1
        #print(a)
    print("dof_position:",env._dof_position)
    deta=env._dof_position-start
    print("deta:",deta)
    
    #print(data_back(env._dof_position,env._robot_limits["joint_position"].low,env._robot_limits["joint_position"].high))

    
# EOF
