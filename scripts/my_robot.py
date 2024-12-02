from isaacgym import gymapi
from leibnizgym.utils import *
import os
from typing import Dict, Tuple, Union, List
import torch
from isaacgym import gymtorch

def set_camera_lookat(gym,viewer, pos: Union[List[float], Tuple[float, float, float]],
                          target: Union[List[float], Tuple[float, float, float]]):
        """Sets the viewer camera position and orientation.

        Args:
            pos: The camera eye's position in world coordinates.
            target: The camera eye's target position in world coordinates.
        """
        cam_pos = gymapi.Vec3(*pos)
        cam_target = gymapi.Vec3(*target)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5
compute_device_id= 0
graphics_device_id=0
physics_engine =gymapi.SIM_PHYSX
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = os.path.join(get_resources_dir(), "assets", "trifinger")
asset_file = "robot_properties_fingers/urdf/edu/trifingeredu.urdf"
robot_asset_options = gymapi.AssetOptions()
robot_asset_options.flip_visual_attachments = False
robot_asset_options.fix_base_link = True
robot_asset_options.collapse_fixed_joints = False
robot_asset_options.disable_gravity = False
robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
robot_asset_options.thickness = 0.001
robot_asset_options.angular_damping = 0.01
robot_asset_options.override_com = True
robot_asset_options.override_inertia = True
robot_asset_options.vhacd_enabled = True
robot_asset_options.vhacd_params.resolution = 300000
robot_asset_options.vhacd_params.max_convex_hulls = 10
robot_asset_options.vhacd_params.max_num_vertices_per_ch = 64

if physics_engine == gymapi.SIM_PHYSX:
    robot_asset_options.use_physx_armature = True
asset = gym.load_asset(sim, asset_root, asset_file,robot_asset_options)

spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)

env = gym.create_env(sim, lower, upper, 8)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, asset, gymapi.Transform(), "MyActor", 0, 1)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
# set camera pose
set_camera_lookat(gym,viewer,pos=(1.0, 1.0, 1.0), target=(0.0, 0.0, 0.0))

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    desired_dof_position=torch.ones(9,dtype=torch.float32,device='cpu')
    gym.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(desired_dof_position))
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

