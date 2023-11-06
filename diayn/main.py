from utils import get_model_and_assets

from dm_control import viewer
from physics import Physics
import numpy as np
from task import Move
from dm_control.rl import control

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Horizontal speeds above which the move reward is 1.
_WALK_SPEED = 0.5


def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  xml_string, assets = get_model_and_assets()
  physics = Physics.from_xml_string(xml_string, assets)
  task = Move(desired_speed=_WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             control_timestep=_CONTROL_TIMESTEP,
                             **environment_kwargs)


env_fn = walk
env = env_fn()

# Get the `action_spec` describing the control inputs.
action_spec = env.action_spec()

# Step through the environment for one episode with random actions.
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(action_spec.minimum, action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  # print("reward = {}, discount = {}, observations = {}.".format(
  #     time_step.reward, time_step.discount, time_step.observation))
  print("reward = {}, discount = {}.".format(
      time_step.reward, time_step.discount))

# random_policy = composer.Policy(lambda time_step: np.random.uniform(action_spec.minimum, action_spec.maximum,
#                              size=action_spec.shape))
# viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_walls)


# policy: An optional callable corresponding to a policy to execute within the
#   environment. It should accept a `TimeStep` and return a numpy array of
#   actions conforming to the output of `environment.action_spec()`.
def sample_random_action(time_step):
  return np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)

viewer.launch(environment_loader=env_fn, policy=sample_random_action)
# viewer.launch(environment_loader=basic_cmu_2019.cmu_humanoid_run_walls, policy=agent.policy)


# from dm_control import suite
# from dm_control import viewer
# import numpy as np

# env = suite.load(domain_name="humanoid", task_name="stand")
# action_spec = env.action_spec()

# # Define a uniform random policy.
# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)
# while not time_step.last():
#   action = np.random.uniform(action_spec.minimum, action_spec.maximum,
#                              size=action_spec.shape)
#   time_step = env.step(action)
#   # print("reward = {}, discount = {}, observations = {}.".format(
#   #     time_step.reward, time_step.discount, time_step.observation))
#   print("reward = {}, discount = {}.".format(
#       time_step.reward, time_step.discount))
