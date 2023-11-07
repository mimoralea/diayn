import argparse
import os
import gym
import torch

import diayn
from diayn.spinningup.spinup.algos.pytorch.sac.sac import core, sac

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAC on the Unitree A1 Gym environment."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="UA1Still-v0",
        help="The name of task to train on.",
    )
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")
    args = parser.parse_args()

    from diayn.spinningup.spinup.utils.run_utils import setup_logger_kwargs

    data_dir = os.path.join(diayn.__path__[0], "..", "data")
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir)

    torch.set_num_threads(torch.get_num_threads())
    sac(
        lambda: gym.make(args.env_id),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
