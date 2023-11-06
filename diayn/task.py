from dm_control.suite import base

from dm_control.rl import control
import numpy as np

from dm_control.utils import rewards
import collections


_STAND_HEIGHT = 1.2
_WALK_SPEED = 1
_RUN_SPEED = 8

class Move(base.Task):

  def __init__(self, desired_speed, pure_state=False, walk=True, random=None):
    self._move_speed = _WALK_SPEED if walk else _RUN_SPEED
    self._desired_speed = desired_speed
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
    upright = (1 + physics.trunk_upright()) / 2
    stand_reward = (3*standing + upright) / 4
    move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                    bounds=(self._move_speed, float('inf')),
                                    margin=self._move_speed/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
    return stand_reward * (5*move_reward + 1) / 6
