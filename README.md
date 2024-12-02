fork from 
# IsaacGym版本的Trifinger安装
IsaacGym中的仿真环境为[GitHub - leibnizgym](https://github.com/pairlab/leibnizgym), 需要用较低版本的[IsaacGym](https://github.com/jmcoholich/isaacgym), 故安装流程如下:
1. 安装isaacgym: `git clone https://github.com/jmcoholich/isaacgym.git`, 按照其中的方法安装IsaacGym, 注意不要安装rl_games版本过低了, 安装完成后可以测试下是否可以gui渲染, 需要将readme下文中注意事项完成, 即添加路径`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib` (也就是`/path/to/mambaforge/envs/rlgpu/lib`文件夹), 让gym能找到动态链接库文件, 一个更好的方法是在`/path/to/mambaforge/envs/rlgpu/etc/conda/activate.d`文件夹下添加一个`*.sh`文件, 里面写上这条命令, 这样就可以启动环境时候自动添加路径了.
2. 安装leibnizgym: `git clone https://github.com/pairlab/leibnizgym.git`进入文件夹后`pip install -e .`
3. 安装1.6.0版本的rl_games: (1.6.1以后的版本需要torch>=2.x) `git clone --branch v1.6.0 --single-branch https://github.com/Denys88/rl_games.git`, 进入文件夹后用`pip install -e .`安装
> 顺序2,3不能颠倒, 因为ieleibnizgym会下载最新版的rl_games, 所以要再降级才能使用
4. (报错`error: invalid value for –gpu-architecture (-arch)`) 需要安装1.13.1版本的pytorch: `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

环境继承关系：IsaacEnvBase -> TrifingerEnv  传入-> VecTaskPython

env_base.py 中 IsaacEnvBase.step 函数

- 先把动作存入action_buf
- 执行pre_step（设置动作执行方法）
- simulate（开始执行物理仿真，执行次数为control_decimation）
- post_step（获取机器人和方块的state，计算reward）

环境和奖励文件：leibnizgym/envs/trifinger/trifinger_env.py, rewards.py

环境创建文件：leibnizgym/utils/rlg_train.py -> RlGameGpuEnvAdapter和create_rl_gpu_env将TrifingerEnv传入到VeecTaskPython

训练文件：scripts/rlg_hydra.py

算法代码：scripts/sac.py, ppo_tt.py...

模型保存：output/{日期} 或者 models/

isaac gym

- urdf文件：resources/assets/trifinger/robot_properties_fingers/urdf/edu/trifingeredu.urdf,
- 网格文件：resources/assets/trifinger/robot_properties_fingers/meshes/...