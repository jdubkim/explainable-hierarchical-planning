import functools

import embodied
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX


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
        # Fields for rendering.
        self._tile_size = 10
        img_sample = self._env.reset()[self._obs_key]
        self._render_shape = self.obs_rendered(img_sample).shape
        self._full_image_shape = (self._env.width, self._env.height, 3)
        self._full_render_shape = (self._env.width * self._tile_size,
                                    self._env.height * self._tile_size, 3)
        self._max_pixel_values = np.array([10, 5, 2])

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
            'full_image': embodied.Space(np.uint8, self._full_image_shape),
            'render': embodied.Space(np.uint8, self._render_shape),
            'full_render': embodied.Space(np.uint8, self._full_render_shape),
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
                   render=self.obs_rendered(obs),
                   full_image=self.full_obs(),
                   full_render=self.full_obs_rendered())
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

    def obs_rendered(self, obs):
        if isinstance(obs, dict):
            obs = obs[self._obs_key]
        return self._env.get_obs_render(obs, tile_size=self._tile_size)
    
    def full_obs(self):
        full_grid = self._env.grid.encode()
        full_grid[self._env.agent_pos[0]][self._env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self._env.agent_dir
        ])

        return full_grid 
    
    def full_obs_rendered(self):
        return self._env.render(
            mode='rgb_array',
            highlight=True,
            tile_size=self._tile_size
        )

    def render_from_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs[self._obs_key]

        # If image is normalised, un-normalise it
        if obs.dtype != np.uint8 and obs.max() <= 1.0:
            obs = (obs * self._max_pixel_values).astype(np.uint8)

        # If single frame, render an image
        if len(obs.shape) == 3:
            return self.obs_rendered(obs)
        # If multiple frames, render each frame and stack them
        elif len(obs.shape) == 4:
            output_shape = (obs.shape[0], ) + self._render_shape
            rendered_obs = np.zeros(output_shape, dtype=np.uint8)
            for i in range(obs.shape[0]):
                rendered_obs[i] = self.obs_rendered(obs[i])
            return rendered_obs

        raise ValueError(f'Invalid observation shape: {obs.shape}')