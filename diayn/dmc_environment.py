import numpy as np
from dm_control.rl import control
from physics import Physics
from utils import get_model_and_assets
from task import Move, TASK_TYPE_TO_MOVE_SPEED


def _dmc_env_creator(task_type, time_limit=20.0, control_timestep=0.02, random=None, environment_kwargs=None):
  """Returns the Walk task."""
  xml_string, assets = get_model_and_assets()
  physics = Physics.from_xml_string(xml_string, assets)
  task = Move(task_type, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                            control_timestep=control_timestep,
                            **environment_kwargs)

ALL_DMC_ENVS = {}
for task_type in TASK_TYPE_TO_MOVE_SPEED:
  ALL_DMC_ENVS[task_type] = lambda tt=task_type: _dmc_env_creator(tt)

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Test a given dmc environment.')
  parser.add_argument('--task', type=str, help='Move task to test: `none`, `still`, `slow`, or `fast`', default='none')
  parser.add_argument('--render', action='store_true', help='Visualize at the end if set')
  args = parser.parse_args()

  assert args.task.lower() in ALL_DMC_ENVS.keys(), f"Unknown task: {args.task}"
  env_fn = ALL_DMC_ENVS[args.task.lower()]
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
    action = random_policy(time_step)
    time_step = env.step(action)
    print("reward = {}, discount = {}.".format(
        time_step.reward, time_step.discount))

  if args.render:
    del env
    from dm_control import viewer
    viewer.launch(environment_loader=env_fn, policy=random_policy)

if __name__ == "__main__":
  main()
