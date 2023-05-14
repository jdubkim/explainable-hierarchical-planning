import functools

import embodied
import numpy as np
import gym
import gym_minigrid


class MiniGrid(embodied.Env):
    def __init__(self, task, obs_key='image', act_key='action'):
        assert task in ('doorkey_flat', 'doorkey_vision')
        self._env = gym.make('MiniGrid-FourRooms-v0')
        self._obs_dict = isinstance(self._env.observation_space.spaces, dict)
        self._act_dict = isinstance(self._env.action_space, dict)
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None
        self._render_shape = self._env.render('rgb_array').shape

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            # Only use the observation space for the specified key.
            spaces = {
                self._obs_key:
                self._env.observation_space.spaces.get(self._obs_key)
            }
            spaces = self._flatten(spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            'render': embodied.Space(np.uint8, self._render_shape),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._env.action_space.spaces.copy()
            spaces = self._flatten(spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return spaces

    def step(self, action):
        if self._done:
            self._done = False
            obs = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, self._done, self._info = self._env.step(action)
        return self._obs(obs,
                         reward,
                         is_last=bool(self._done),
                         is_terminal=bool(
                             self._info.get('is_terminal', self._done)))

    def _obs(self,
             obs,
             reward,
             is_first=False,
             is_last=False,
             is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        else:
            obs = {self._obs_key: obs.get(self._obs_key, {})}
        obs = self._flatten(obs)
        obs = {k: np.array(v) for k, v in obs.items()}
        obs.update(reward=np.float32(reward),
                   is_first=is_first,
                   is_last=is_last,
                   is_terminal=is_terminal,
                   render=self.render_from_obs(obs))
        return obs

    def render(self):
        return self._env.render('rgb_array')

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        """
    Flatten a nested dictionary into a single dictionary. 
    """
        result = {}
        for key, value in nest.items():
            key = f'{prefix}/{key}' if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        """
    Unflatten a dictionary into a nested dictionary. 
    """
        result = {}
        for key, value in flat.items():
            parts = key.split('/')
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, 'n'):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)

    def render_from_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs[self._obs_key]
        if len(obs.shape) == 3:
            return self._env.get_obs_render(obs)

        # If multiple frames, loop through and render each one
        if len(obs.shape) == 4:
            output_shape = (obs.shape[0], ) + self._render_shape
            obs = np.zeros(output_shape, dtype=np.uint8)
            for i in range(obs.shape[0]):
                obs[i] = self._env.get_obs_render(obs[i])

            return obs
