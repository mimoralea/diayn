import argparse

import gym
from diayn.environments import diayn


def main(task="none"):
    # Create the Gym environment based on the task string
    env = gym.make(task)

    # Set the number of episodes to run
    num_episodes = 10

    # Run the reinforcement learning loop for the specified number of episodes
    for i in range(num_episodes):
        # Reset the environment for each episode
        obs = env.reset()
        done = False
        total_reward = 0

        # Run the episode until termination
        while not done:
            # Choose a random action
            action = env.action_space.sample()

            # Take the chosen action and observe the next state and reward
            obs, reward, done, info = env.step(action)

            # Update the total reward for the episode
            total_reward += reward

        # Print the total reward for the episode
        print(f"Task: {task}, Episode {i+1}: Total reward = {total_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAC on the Unitree A1 Gym environment."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="none",
        help="The name of task to train on.",
    )
    args = parser.parse_args()

    main(args.task)

if __name__ == "__main__":
    main()
