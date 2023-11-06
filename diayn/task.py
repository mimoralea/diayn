from dm_control.suite import base
from utils import get_model_and_assets
from enum import Enum

from physics import Physics
from dm_control.rl import control
import numpy as np

from dm_control.utils import rewards
import collections


_STAND_HEIGHT = 1.2

class Speed(Enum):
    STILL = 0
    WALK = 1
    RUN = 8

class Move(base.Task):

  def __init__(self, pure_state=False, speed=Speed.STILL, random=None):
    self._move_speed = speed.value
    self._pure_state = pure_state
    super().__init__(random=random)

  def initialize_episode(self, physics):
    orientation = self.random.randn(4)
    orientation /= np.linalg.norm(orientation)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    obs = collections.OrderedDict()
    if self._pure_state:
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
    else:
        obs['trunk_upright'] = physics.trunk_upright()
        obs['joint_angles'] = physics.joint_angles()
        obs['trunk_vertical'] = physics.trunk_vertical_orientation()
        obs['com_position'] = physics.center_of_mass_position()
        obs['com_velocity'] = physics.center_of_mass_velocity()
        obs['velocity'] = physics.velocity()

    return obs

  def get_reward(self, physics):
    standing = rewards.tolerance(physics.trunk_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/2)
    upright = rewards.tolerance(physics.trunk_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
    stand_reward = (3*standing + upright) / 4
    if self._move_speed == 0:
      horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
      dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
      return stand_reward * dont_move
    else:

      move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                      bounds=(self._move_speed, float('inf')),
                                      margin=self._move_speed/2,
                                      value_at_margin=0.5,
                                      sigmoid='linear')

      return stand_reward * (5*move_reward + 1) / 6


def walk(time_limit=20.0, control_timestep=0.02, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  xml_string, assets = get_model_and_assets()
  physics = Physics.from_xml_string(xml_string, assets)
  task = Move(speed=Speed.WALK, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                            control_timestep=control_timestep,
                            **environment_kwargs)


def run(time_limit=20.0, control_timestep=0.02, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  xml_string, assets = get_model_and_assets()
  physics = Physics.from_xml_string(xml_string, assets)
  task = Move(speed=Speed.RUN, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                            control_timestep=control_timestep,
                            **environment_kwargs)


def standup(time_limit=20.0, control_timestep=0.02, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  xml_string, assets = get_model_and_assets()
  physics = Physics.from_xml_string(xml_string, assets)
  task = Move(speed=Speed.STILL, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                            control_timestep=control_timestep,
                            **environment_kwargs)

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Test a given task.')
  parser.add_argument('task', type=str, help='Task to test: `walk`, `run`, or `standup`', default='standup')
  parser.add_argument('--visualize', action='store_true', help='Visualize at the end if set')
  args = parser.parse_args()

  env_fn = None
  if args.task.lower() == 'walk':
    env_fn = walk
  elif args.task.lower() == 'run':
    env_fn = run
  elif args.task.lower() == 'standup':
    env_fn = standup
  else:
    NotImplementedError(f"Unknown task: {args.task}")

  env = env_fn()
  action_spec = env.action_spec()
  min_action = action_spec.minimum
  max_action = action_spec.maximum
  def random_policy(_):
    return np.random.uniform(
      min_action,
      max_action
    )

  time_step = env.reset()
  while not time_step.last():
    action = np.random.uniform(action_spec.minimum, action_spec.maximum,
                              size=action_spec.shape)
    time_step = env.step(action)
    # print("reward = {}, discount = {}, observations = {}.".format(
    #     time_step.reward, time_step.discount, time_step.observation))
    print("reward = {}, discount = {}.".format(
        time_step.reward, time_step.discount))

  if args.visualize:
    del env
    from dm_control import viewer
    viewer.launch(environment_loader=env_fn, policy=random_policy)

if __name__ == "__main__":
  main()