from gym.envs import register

from diayn.environments.ua1_gym import ALL_GYM_ENVS

for name, env_fn in ALL_GYM_ENVS.items():
    env_id = f'UA1{name.capitalize()}-v0'
    register(
        id=env_id,
        entry_point=env_fn
    )
