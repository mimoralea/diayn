import gym
from gym import spaces
import numpy as np

from dm_control import suite
from dm_env import specs
from task import ALL_TASKS


def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space. """
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum, 
                           high=dm_control_space.maximum, 
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=-float('inf'), 
                           high=float('inf'), 
                           shape=dm_control_space.shape, 
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space

class DMSuiteEnv(gym.Env):
    def __init__(self, task, task_kwargs=None):
        self.env = task(**(task_kwargs or {}))
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None
    
    def seed(self, seed):
        return self.env.task.random.seed(seed)
    
    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        timestep = self.env.reset()
        return timestep.observation
    
    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 0  # Tracking camera
        use_opencv_renderer = kwargs.pop('use_opencv_renderer', True)
        
        img = self.env.physics.render(**kwargs)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                if not use_opencv_renderer:
                    from gym.envs.classic_control import rendering
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1024)
                else:
                    from utils import OpenCVImageViewer
                    self.viewer = OpenCVImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()

ALL_ENVS = {}
for name, task in ALL_TASKS.items():
    ALL_ENVS[f'UA1{name.capitalize()}-v0'] = lambda **kwargs: DMSuiteEnv(task, **kwargs)

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Test a given environment.')
  parser.add_argument('--env_id', type=str, help='Unitree A1 environment to test: `UA1Zero-v0`, `UA1Still-v0`, `UA1Slow-v0`, or `UA1Fast-v0`', default='UA1Zero-v0')
  parser.add_argument('--render', action='store_true', help='Render in the RL loop')
  args = parser.parse_args()

  assert args.env_id in ALL_ENVS.keys(), f"Unknown environment: {args.env_id}"
  env_fn = ALL_ENVS[args.env_id]
  env = env_fn()

  observation, done = env.reset(), False
  args.render and env.render()
  while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("reward = {}, done = {}.".format(
        reward, done))
    args.render and env.render()

if __name__ == "__main__":
  main()
