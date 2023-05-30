import functools
import os

import embodied
import numpy as np


class DMEnv(embodied.Env):

  def __init__(self, env):
    self._env = env
    self._obs_dict = isinstance(self._env.observation_spec(), dict)
    self._act_dict = isinstance(self._env.action_spec(), dict)
    self._obs_key = not self._obs_dict and self._env.observation_spec().name
    self._act_key = not self._act_dict and self._env.action_spec().name
    self._done = True

  @functools.cached_property
  def obs_space(self):
    spec = self._env.observation_spec()
    spec = spec if self._obs_dict else {spec.name: spec}
    if "reward" in spec:
      spec["obs_reward"] = spec.pop("reward")
    return {
        "reward": embodied.Space(np.float32),
        "is_first": embodied.Space(bool),
        "is_last": embodied.Space(bool),
        "is_terminal": embodied.Space(bool),
        **{k: self._convert(v) for k, v in spec.items()},
    }

  @functools.cached_property
  def act_space(self):
    spec = self._env.action_spec()
    spec = spec if self._act_dict else {spec.name: spec}
    return {
        "reset": embodied.Space(bool),
        **{k: self._convert(v) for k, v in spec.items()},
    }

  def step(self, action):
    action = action.copy()
    reset = action.pop("reset")
    if reset or self._done:
      time_step = self._env.reset()
    else:
      action = action if self._act_dict else action[self._act_key]
      time_step = self._env.step(action)
    self._done = time_step.last()
    return self._obs(time_step)

  def _obs(self, time_step):
    if not time_step.first():
      assert time_step.discount in (0, 1), time_step.discount
    obs = time_step.observation
    obs = obs if self._obs_dict else {self._obs_key: obs}
    if "reward" in obs:
      obs["obs_reward"] = obs.pop("reward")
    return dict(
        reward=0.0 if time_step.first() else time_step.reward,
        is_first=time_step.first(),
        is_last=time_step.last(),
        is_terminal=False if time_step.first() else time_step.discount == 0,
        **dict(obs),
    )

  def _convert(self, space):
    if hasattr(space, "num_values"):
      return embodied.Space(space.dtype, (), 0, space.num_values)
    else:
      return embodied.Space(space.dtype, space.shape, space.minimum, space.maximum)
