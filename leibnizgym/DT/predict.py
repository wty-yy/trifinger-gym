from leibnizgym.DT.dt_model import GPT
from flax.training import train_state
import jax
import numpy as np
from pathlib import Path

path_root = Path(__file__).parents[0]
weights_name = "DT_tri__0__20240503_172032/010.npy"
# weights_name = "DT_tri_v4__0__mydata_v4_1367861__20240507_211258/ckpt/010.npy"
# weights_name = "DT_tri__0__mydata_v3_663750_all__20240503_204651/010.npy"
path_weights = path_root / 'weights' / weights_name


from leibnizgym.DT.dt_model import GPTConfig, GPT, TrainConfig
class Predictor:
  def __init__(self, path_weights=path_weights, load_step=None, rtg=20, seed=42):
    self.base_rtg, self.seed = rtg, seed
    self._load_model(path_weights, load_step)
    self.n_step = self.model.cfg.n_token // 3
    self.action_dim = self.model.cfg.act_dim
    self.reset()

  def _load_model(self, path_weights, load_step):
    if path_weights.suffix == '.npy':
      load_info = np.load(path_weights, allow_pickle=True).item()
    else:
      ckpt_mngr = CheckpointManager(path_weights)
      if load_step is None:
        load_step = int(sorted(path_weights.glob('*'))[-1].stem)
      load_info = ckpt_mngr.restore(load_step)
    # np.save(path_weights, load_info, allow_pickle=True)
    params, cfg = load_info['params'], load_info['config']
    self.model = GPT(cfg=GPTConfig(**cfg))
    self.model.create_fns()
    state = self.model.get_state(TrainConfig(**cfg), train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)

  def _get_action(self, deterministic: bool):
    n_step = self.n_step
    def pad(x):
      delta = max(n_step - len(x), 0)
      x = np.expand_dims(np.stack(x[-n_step:]), 0)
      if x.ndim == 2:
        return np.pad(x, ((0, 0), (0, delta)))
      else:  # state: (1, N, 20) or action: (1, N, 4)
        return np.pad(x, ((0, 0), (0, delta), (0, 0)))
    mask_len = np.array([min(3 * len(self.s) - 1, n_step * 3 - 1)], np.int32)  # the last action is awalys padding
    rng, self.rng = jax.random.split(self.rng)
    action = jax.device_get(self.model.predict(
      self.state,
      pad(self.s).astype(np.float32),
      pad(self.a).astype(np.int32),
      pad(self.rtg).astype(np.float32),
      pad(self.timestep).astype(np.int32),
      mask_len, rng, deterministic))
    return action[0]
  
  def reset(self):
    self.rng = jax.random.PRNGKey(self.seed)
    self.time_count = 1
    self.s, self.a, self.rtg, self.timestep = [], [np.zeros(self.action_dim)], [self.base_rtg], [1]
    self.score = 0
  
  def __call__(self, state, last_reward=None, deterministic=False):
    self.s.append(state)
    if self.timestep != 1:
      assert last_reward is not None, "Please give your last_reward return by the env"
      self.rtg.append(max(self.rtg[-1] - last_reward, 1))
      self.score += last_reward
    a = self._get_action(deterministic)
    self.a[-1] = a; self.a.append(np.zeros(self.action_dim))
    self.time_count = min(self.time_count + 1, self.model.cfg.max_timestep)
    self.timestep.append(self.time_count)
    return a

if __name__ == '__main__':
  predictor = Predictor(rtg=1.5)
  np.random.seed(True)
  for i in range(10):
    s = np.random.rand(41)
    a = predictor(s, 0.1)
    print(a.shape)
    print(a)
