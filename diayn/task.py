import collections

import numpy as np
from dm_control.suite import base
from dm_control.utils import rewards

_TARGET_HEIGHT = 0.3

TASK_TO_SPEED = {"none": -1, "still": 0, "slow": 2, "fast": 8}


class Move(base.Task):
    def __init__(self, task_type, pure_state_observations=False, random=None):
        self._target_speed = TASK_TO_SPEED[task_type]
        self._pure_state_observations = pure_state_observations
        super().__init__(random=random)

    def initialize_episode(self, physics):
        orientation = self.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        if self._pure_state_observations:
            obs["position"] = physics.position()
            obs["velocity"] = physics.velocity()
        else:
            obs["trunk_upright"] = physics.trunk_upright()
            obs["joint_angles"] = physics.joint_angles()
            obs["trunk_vertical"] = physics.trunk_vertical_orientation()
            obs["com_position"] = physics.center_of_mass_position()
            obs["com_velocity"] = physics.center_of_mass_velocity()
            obs["velocity"] = physics.velocity()

        return obs

    def get_reward(self, physics):
        if self._target_speed == -1:
            return 0.0

        standing = rewards.tolerance(
            physics.trunk_height(),
            bounds=(_TARGET_HEIGHT, float("inf")),
            margin=_TARGET_HEIGHT / 2,
        )
        # upright = rewards.tolerance(
        #     physics.trunk_upright(),
        #     bounds=(0.9, float("inf")),
        #     sigmoid="linear",
        #     margin=0.8,
        #     value_at_margin=0,
        # )
        # stand_reward = (9 * standing + upright) / 10
        if self._target_speed == 0:
            return standing

        if self._target_speed == 0:
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return stand_reward * dont_move

        move_reward = rewards.tolerance(
            physics.horizontal_velocity(),
            bounds=(self._target_speed, float("inf")),
            margin=self._target_speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )

        return (2 * stand_reward + move_reward) / 3
