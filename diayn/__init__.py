from gym.envs import register
from dm_control import suite
from environment import ALL_A1_ENVS

for name, task in ALL_A1_ENVS:
    ID = f'{name.capitalize()}{task.capitalize()}-v0'
    register(id=ID, 
             entry_point='diayn.environment:DMSuiteEnv')
