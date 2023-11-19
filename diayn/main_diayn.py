import argparse
import os
import gym
import torch
import dmc2gym

import diayn as diayn_m
from diayn.spinningup.spinup.algos.pytorch.diayn.diayn import core, diayn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DIAYN on the Unitree A1 Gym environment."
    )
    parser.add_argument("--exp_name", type=str, default="diayn")
    parser.add_argument(
        "--env_id",
        type=str,
    )
    parser.add_argument(
        "--domain_name",
        type=str,
    )
    parser.add_argument(
        "--task_name",
        type=str,
    )

    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--n_skill", type=int, default=20)
    parser.add_argument("--curriculum_threshold", type=float, default=0.7)
    parser.add_argument("--task_min", type=float, default=0.0)
    parser.add_argument("--task_max", type=float, default=2.0)
    parser.add_argument("--intrinsic_min", type=float, default=1.0)
    parser.add_argument("--intrinsic_max", type=float, default=5.0)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    # steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
    # polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
    # update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    # hidden_sizes=(256,256),
    # activation=nn.ReLU
    args = parser.parse_args()
    # domain_name='walker', task_name='walk'
    assert args.env_id or (
        args.domain_name and args.task_name
    ), "Can't create environment"

    from diayn.spinningup.spinup.utils.run_utils import setup_logger_kwargs

    env_folder = args.env_id if args.env_id else f"{args.domain_name}_{args.task_name}"
    data_dir = os.path.join(diayn_m.__path__[0], "..", "data", env_folder)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir)

    env_fn = (
        lambda: gym.make(args.env_id)
        if args.env_id
        else dmc2gym.make(
            domain_name=args.domain_name, task_name=args.task_name, seed=args.seed
        )
    )

    torch.set_num_threads(torch.get_num_threads())
    diayn(
        env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        n_skill=args.n_skill,
        curriculum_threshold=args.curriculum_threshold,
        task_min=args.task_min,
        task_max=args.task_max,
        intrinsic_min=args.intrinsic_min,
        intrinsic_max=args.intrinsic_max,
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        logger_kwargs=logger_kwargs,
    )
