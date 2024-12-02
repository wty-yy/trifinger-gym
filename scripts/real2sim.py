from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
from leibnizgym.utils.torch_utils import saturate, unscale_transform, scale_transform, quat_diff_rad
# python
import torch
import numpy as np
from isaacgym import gymtorch
import sys
from tcp import *
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
if __name__=='__main__':
    client=tcp_init()
    env = TrifingerEnv(config=env_config, device='cpu', verbose=True, visualize=True)
    _ = env.reset()
    print_info("Trifinger environment creation successful.")
    instances=0
    import time
    while True:
        get_message(client)
        env._gym.refresh_dof_state_tensor(env._sim)
        env._gym.refresh_actor_root_state_tensor(env._sim)
        env._gym.refresh_rigid_body_state_tensor(env._sim)

        goal_object_indices = env._gym_indices["goal_object"][instances]
        #env._actors_root_state[goal_object_indices, 0:7] = torch.tensor([0.1,0.1,0.033,0,0,0,1],dtype=torch.float32)
        env._actors_root_state[goal_object_indices, 0:7] = trifinger.target_cube_state
        
        goal_object_indices = env._gym_indices["goal_object"].to(torch.int32)
        env._gym.set_actor_root_state_tensor_indexed(env._sim, gymtorch.unwrap_tensor(env._actors_root_state),
                                                            gymtorch.unwrap_tensor(goal_object_indices), len(goal_object_indices))
        object_indices = env._gym_indices["object"][instances]
        # # set values into buffer
        # # object buffer
        # env._object_state_history[0][instances, 0] = 0
        # env._object_state_history[0][instances, 1] = 0
        # env._object_state_history[0][instances, 2] = 0.0325
        # env._object_state_history[0][instances, 3:7] = torch.tensor([0,0,0,1],dtype=torch.float32)
        env._object_state_history[0][instances, 0:7] = trifinger.cube_state
        env._object_state_history[0][instances, 7:13] = 0
        env._actors_root_state[object_indices] = env._object_state_history[0][instances]
        object_indices = env._gym_indices["object"].to(torch.int32)
        env._gym.set_actor_root_state_tensor_indexed(env._sim, gymtorch.unwrap_tensor(env._actors_root_state),
                                                            gymtorch.unwrap_tensor(object_indices), len(object_indices))
        # desired_dof_position=torch.tensor([0,0.9,-1.7]*3,dtype=torch.float32)
        desired_dof_position=trifinger.dof_pos
        env._gym.set_dof_position_target_tensor(env._sim,gymtorch.unwrap_tensor(desired_dof_position))
        env._gym.simulate(env._sim)
        env._gym.fetch_results(env._sim, True)
        env.render()
        time.sleep(0.1)
