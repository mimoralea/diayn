from gym.envs import register
from dm_control import suite
from diayn.gym_environment import ALL_A1_ENVS

# f'UA1{name.capitalize()}-v0'
for name, task in ALL_A1_ENVS:
    ID = f"{name.capitalize()}{task.capitalize()}-v0"
    register(id=ID, entry_point="diayn.environment:DMSuiteEnv")
