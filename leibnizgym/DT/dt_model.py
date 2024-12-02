import jax, jax.numpy as jnp
import flax.linen as nn
import flax, optax
import numpy as np
from flax.training import train_state
from typing import Callable, Sequence
from functools import partial

class Config:
  def __iter__(self):
    for name in dir(self):
      val = getattr(self, name)
      if '__' not in name:
        yield (name, val)

  def __repr__(self):
    return str(dict(self))

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    
Dense = partial(nn.Dense, kernel_init=nn.initializers.normal(stddev=0.02))
Embed = partial(nn.Embed, embedding_init=nn.initializers.normal(stddev=0.02))

class TrainState(train_state.TrainState):
  dropout_rng: jax.Array

class GPTConfig(Config):
  act_dim = 9
  n_embd = 128  # 768
  n_head = 8  # 12
  n_block = 6  # 12
  p_drop_embd = 0.1
  p_drop_resid = 0.1
  p_drop_attn = 0.1

  def __init__(self, n_token, max_timestep, **kwargs):
    self.n_token, self.max_timestep = n_token, max_timestep
    for k, v in kwargs.items():
      setattr(self, k, v)
    assert self.n_embd % self.n_head == 0, "n_embd must be devided by n_head"

class TrainConfig(Config):
  obs_dim = 41
  seed = 42
  weight_decay = 1e-4  # 0.1
  lr = 6e-4
  total_epochs = 5
  batch_size = 128
  betas = (0.9, 0.95)  # Adamw beta1, beta2
  # warmup_tokens = 128*128*256  # 375e6
  warmup_tokens = 512*20  # 375e6
  clip_global_norm = 1.0
  lr_fn: Callable

  def __init__(self, steps_per_epoch, n_token, **kwargs):
    self.steps_per_epoch = steps_per_epoch
    self.n_token = n_token
    for k, v in kwargs.items():
      setattr(self, k, v)

class CausalSelfAttention(nn.Module):
  n_embd: int  # NOTE: n_embd % n_head == 0
  n_head: int
  p_drop_attn: float

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool):
    D = self.n_embd // self.n_head  # hidden dim
    B, L, _ = x.shape  # Bachsize, token length, embedding dim
    mask = jnp.expand_dims(jnp.tri(L), (0, 1))  # Only consider previous token values
    x = Dense(3 * self.n_embd)(x)
    q, k, v = jnp.array_split(x.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3), 3, -1)
    attn = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(D)
    attn = jnp.where(mask == 0, -1e18, attn)
    attn = jax.nn.softmax(attn)
    attn = nn.Dropout(self.p_drop_attn)(attn, deterministic=not train)
    y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.n_embd)
    y = jnp.tanh(Dense(self.n_embd)(y))
    return y

class AttentionBlock(nn.Module):
  cfg: GPTConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool):
    attn_cfg = {key: getattr(self.cfg, key) for key in ['n_embd', 'n_head', 'p_drop_attn']}
    z = nn.LayerNorm()(x)
    z = CausalSelfAttention(**attn_cfg)(z, train)
    x = x + nn.Dropout(self.cfg.p_drop_resid)(z, deterministic=not train)
    z = nn.Sequential([
      nn.LayerNorm(),
      Dense(4*self.cfg.n_embd), nn.gelu,
      Dense(self.cfg.n_embd),
    ])(x)
    x = x + nn.Dropout(self.cfg.p_drop_resid)(z, deterministic=not train)
    return x

class GPT(nn.Module):
  cfg: GPTConfig

  @nn.compact
  def __call__(self, s, a, rtg, timestep, train: bool):
    cfg = self.cfg
    B, l = rtg.shape
    assert cfg.n_token == l * 3, "The n_token should be 3 * n_step"
    ### Embedding ###
    rtg = nn.tanh(Dense(cfg.n_embd)(jnp.expand_dims(rtg, -1)))  # (B, l) -> (B, l, N_e)
    s = nn.tanh(Dense(cfg.n_embd)(s))  # (B, l, 20) -> (B, l, N_e)
    a = nn.tanh(Dense(cfg.n_embd)(a))  # (B, l, 9) -> (B, l, N_e)
    time_embd = nn.Embed(cfg.max_timestep+1, cfg.n_embd, embedding_init=nn.initializers.zeros)(timestep)  # (B, l) -> (B, l, N_e)
    pos_embd = nn.Embed(cfg.n_token, cfg.n_embd, embedding_init=nn.initializers.zeros)(jnp.arange(cfg.n_token))  # (1, L, N_e)
    ### Build Token ###
    x = jnp.concatenate([jnp.expand_dims(x, 2) for x in [rtg, s, a]], 2).reshape(B, cfg.n_token, cfg.n_embd)  # (B, l, 3, N_e) -> (B, L, N_e)
    x = x + pos_embd + time_embd.repeat(3, 1)  # (B, L, N_e)
    ### GPT-1 ###
    x = nn.Dropout(cfg.p_drop_embd)(x, deterministic=not train)
    for _ in range(cfg.n_block):
      x = AttentionBlock(cfg)(x, train)
    x = nn.LayerNorm()(x)
    x = Dense(cfg.act_dim, use_bias=False)(x)
    return x
    
  def get_state(self, train_cfg: TrainConfig, verbose: bool = False, load_path: str = None, train: bool = True) -> TrainState:
    def check_decay_params(kp, x):
      fg = x.ndim > 1
      for k in kp:
        if k.key in ['LayerNorm', 'Embed']:
          fg = False; break
      return fg
    def lr_fn():
      warmup_steps = train_cfg.warmup_tokens // (train_cfg.n_token * train_cfg.batch_size)
      warmup_fn = optax.linear_schedule(0.0, train_cfg.lr, warmup_steps)
      second_steps = max(train_cfg.total_epochs * train_cfg.steps_per_epoch - warmup_steps, 1)
      second_fn = optax.cosine_decay_schedule(
        train_cfg.lr, second_steps, 0.1
      )
      return optax.join_schedules(
        schedules=[warmup_fn, second_fn],
        boundaries=[warmup_steps]
      )
    rng = jax.random.PRNGKey(train_cfg.seed)
    if not train:  # return state with apply function
      return TrainState.create(apply_fn=self.apply, params={'a': 1}, tx=optax.sgd(1), dropout_rng=rng)
    # s, a, rtg, timestep
    B, l = train_cfg.batch_size, self.cfg.n_token // 3
    s, a, rtg, timestep = jnp.empty((B, l, train_cfg.obs_dim), float), jnp.empty((B, l, self.cfg.act_dim), int), jnp.empty((B, l), float), jnp.empty((B, l), int)
    examp = (s, a, rtg, timestep)
    if verbose: print(self.tabulate(rng, *examp, train=False))
    variables = self.init(rng, *examp, train=False)
    print("mini-GPT params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variables)[0]]))
    decay_mask = jax.tree_util.tree_map_with_path(check_decay_params, variables['params'])
    train_cfg.lr_fn = lr_fn()
    state = TrainState.create(
      apply_fn=self.apply,
      params=variables['params'],
      # AdamW is Adam with weight decay
      tx=optax.chain(
        optax.clip_by_global_norm(train_cfg.clip_global_norm),
        optax.adamw(train_cfg.lr_fn, train_cfg.betas[0], train_cfg.betas[1], weight_decay=train_cfg.weight_decay, mask=decay_mask),
      ),
      dropout_rng=rng,
    )
    if load_path is not None:
      with open(load_path, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
      print(f"Load weights from {load_path}")
    return state
  
  def create_fns(self):
    def model_step(state: TrainState, s, a, rtg, timestep, y, train: bool):
      dropout_rng, base_rng = jax.random.split(state.dropout_rng)
      def loss_fn(params):
        logits = state.apply_fn({'params': params}, s, a, rtg, timestep, train=train, rngs={'dropout': dropout_rng})
        logits = logits[:, 1::3, :]  # (B, l, N_e)
        loss = (0.5 * (logits - a) ** 2).mean()
        return loss
      loss, grads = jax.value_and_grad(loss_fn)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(dropout_rng=base_rng)
      return state, loss
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, s, a, rtg, timestep, mask_len: Sequence[int] = None, rng: jax.Array = None, deterministic: bool = False):
      # print(s.shape, a.shape, rtg.shape, timestep.shape, mask_len.shape)
      logits = state.apply_fn({'params': state.params}, s, a, rtg, timestep, train=False)
      # logits = logits[:, 1::3, :]  # (B, l, N_e)
      if mask_len is not None:
        pred = logits[jnp.arange(logits.shape[0]), mask_len-1, :]  # (B, pred_actions)
      return pred
    self.predict = jax.jit(predict, static_argnames='deterministic')

  def save_model(self, state, save_path):
    with open(save_path, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    print(f"Save weights to {save_path}")
  
if __name__ == '__main__':
  batch_size = 128
  n_token = 90
  n_embd = 128
  n_head = 8
  n_block = 6
  max_timestep = 3000
  # Total Parameters: 1,590,400 (6.4 MB)
  gpt_cfg = GPTConfig(n_token, max_timestep=max_timestep, n_embd=n_embd, n_head=n_head, n_block=n_block)
  print(dict(gpt_cfg))
  gpt = GPT(gpt_cfg)
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.randint(rng, (batch_size, n_len), 0, 6)
  # print(gpt.tabulate(rng, x, train=False))
  # variable = gpt.init(rng, x, train=False)
  # print("params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variable)[0]]))
  train_cfg = TrainConfig(steps_per_epoch=512, n_token=n_token)
  state = gpt.get_state(train_cfg, verbose=True)
