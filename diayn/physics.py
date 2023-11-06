from dm_control import mujoco
import numpy as np

class Physics(mujoco.Physics):

  def _reload_from_data(self, data):
    super()._reload_from_data(data)
    self._hinge_names = []

  def trunk_height(self):
    return self.named.data.xpos['trunk', 'z']

  def trunk_upright(self):
    return np.asarray(self.named.data.xmat['trunk', 'zz'])

  def trunk_vertical_orientation(self):
    return self.named.data.xmat['trunk', ['zx', 'zy', 'zz']]

  def center_of_mass_position(self):
    return self.named.data.subtree_com['trunk'].copy()

  def center_of_mass_velocity(self):
    return self.named.data.sensordata['trunk_subtreelinvel']

  def horizontal_velocity(self):
    return self.named.data.sensordata['trunk_subtreelinvel'][0]

  def joint_angles(self):
    return self.data.qpos[7:].copy()  # Skip the 7 DoFs of the free root joint.
