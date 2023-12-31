import gym
from dm_env import specs
from diayn.environments.ua1_dmc import ALL_DMC_ENVS
from diayn.utils import OpenCVImageViewer



def convert_dm_control_to_gym_space(dm_control_space):
    r"""Convert dm_control space to gym space."""
    if isinstance(dm_control_space, specs.BoundedArray):
        space = gym.spaces.Box(
            low=dm_control_space.minimum,
            high=dm_control_space.maximum,
            dtype=dm_control_space.dtype,
        )
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(
        dm_control_space, specs.BoundedArray
    ):
        space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=dm_control_space.dtype,
        )
        return space
    elif isinstance(dm_control_space, dict):
        space = gym.spaces.Dict(
            {
                key: convert_dm_control_to_gym_space(value)
                for key, value in dm_control_space.items()
            }
        )
        # TODO: Fix this hacky way to get the observation
        assert "observations" in space.spaces, "'observations' must be present in Dict space, not other structure supported"
        return space["observations"]


class DMSuiteEnv(gym.Env):
    def __init__(self, task, task_kwargs=None):
        self.env = task(**(task_kwargs or {}))
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": round(1.0 / self.env.control_timestep()),
        }

        self.observation_space = convert_dm_control_to_gym_space(
            self.env.observation_spec()
        )
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        self.viewer = None

    def seed(self, seed):
        return self.env.task.random.seed(seed)

    def step(self, action):
        timestep = self.env.step(action)
        # TODO: Fix this hacky way to get the observation
        observation = timestep.observation["observations"]
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        # TODO: Fix this hacky way to get the observation
        return timestep.observation["observations"]

    def render(self, mode="human", **kwargs):
        if "camera_id" not in kwargs:
            kwargs["camera_id"] = 0  # Tracking camera

        img = self.env.physics.render(**kwargs)
        if self.viewer is None and mode=="human":
            self.viewer = OpenCVImageViewer()

        if self.viewer:
            self.viewer.imshow(img)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()


ALL_GYM_ENVS = {}
for name, dmc_env in ALL_DMC_ENVS.items():
    ALL_GYM_ENVS[name] = lambda de=dmc_env, **kwargs: DMSuiteEnv(de, **kwargs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test a given environment.")
    parser.add_argument(
        "--task",
        type=str,
        help="Move task to test: `none`, `still`, `slow`, or `fast`",
        default="none",
    )
    parser.add_argument(
        "--render", action="store_true", help="Visualize inside the render loop"
    )
    args = parser.parse_args()

    assert args.task in ALL_GYM_ENVS.keys(), f"Unknown environment: {args.task}"
    env_fn = ALL_GYM_ENVS[args.task]
    env = env_fn()

    _, done = env.reset(), False
    args.render and env.render()
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        print("reward = {}, done = {}.".format(reward, done))
        args.render and env.render()


if __name__ == "__main__":
    main()
