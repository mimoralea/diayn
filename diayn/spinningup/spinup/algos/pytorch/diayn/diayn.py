from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
import diayn.spinningup.spinup.algos.pytorch.diayn.core as core
from diayn.spinningup.spinup.utils.logx import EpochLogger

EPS = torch.as_tensor(1E-6, dtype=torch.float32)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DIAYN agents.
    """

    def __init__(self, sk_dim, obs_dim, act_dim, size):
        self.sk_buf = np.zeros(core.combined_shape(size, sk_dim), dtype=np.float32)
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.irew_buf = np.zeros(size, dtype=np.float32)
        self.wrew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sk, obs, act, rew, irew, wrew, next_obs, done):
        self.sk_buf[self.ptr] = sk
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.irew_buf[self.ptr] = irew
        self.wrew_buf[self.ptr] = wrew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            sk=self.sk_buf[idxs],
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            irew=self.irew_buf[idxs],
            wrew=self.wrew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def diayn(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    n_skill=20,
    curriculum_threshold=0.7,
    task_min=0.0,
    task_max=2.0,
    intrinsic_min=1.0,
    intrinsic_max=5.0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=1,
):
    """


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to diayn.

        seed (int): Seed for random number generators.

        n_skill (int): Number of skills to learn

        intrinsic_w (float): Importance ratio between task and intrinsic reward 

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(n_skill, env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Helper function to get a one-hot encoded skill vector
    g_sk = lambda n: np.eye(n, k=np.random.randint(n))[0]

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(sk_dim=n_skill, obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # check right values
    assert 0 <= curriculum_threshold <= 1, f"Task threshold must be 0...1, got {curriculum_threshold}"
    assert 0 <= task_min <= 10, f"Task min must be 0...10, got {task_min}"
    assert 0 <= task_max <= 10, f"Task max must be 0...10, got {task_max}"
    assert 0 <= intrinsic_min <= 10, f"Intrinsic min must be 0...10, got {intrinsic_min}"
    assert 0 <= intrinsic_max <= 10, f"Intrinsic max must be 0...10, got {intrinsic_max}"

    def get_discriminator_confidence(s, o):
        assert len(s.shape) == len(o.shape) == 1, "Function can't handle batches"

        # Convert to tensors
        s = torch.as_tensor(s, dtype=torch.float32)
        o = torch.as_tensor(o, dtype=torch.float32)

        # Get logits from discriminator net
        d = ac_targ.di(o)

        # Convert to probs
        p_s = max(d.softmax(0)[s.argmax()], EPS)

        # Disc probability of skill
        return p_s.item()

    def compute_weighted_reward(dc, tp):

        # (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        rescale = lambda x, min, max: x * (max - min) + min

        # rescale the task reward
        tr = rescale(tp, task_min, task_max)
        
        # check curriculum
        # if per-timestep task reward is greater or equal to threshold,
        # move to next phase of curriculum. tp: 0...1
        if tp < curriculum_threshold:
            return tr

        # rescale the intrinsic reward
        im = rescale(dc, intrinsic_min, intrinsic_max)

        # augment the task reward (tp -> tr) based on
        # the discriminator confidence (dc -> im)
        # only after agent dominates task (threshold)
        return tr * im

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        s, o, a, r, ir, wr, o2, d = (
            data["sk"],
            data["obs"],
            data["act"],
            data["rew"],
            data["irew"],
            data["wrew"],
            data["obs2"],
            data["done"],
        )

        q1 = ac.q1(s, o, a)
        q2 = ac.q2(s, o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(s, o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(s, o2, a2)
            q2_pi_targ = ac_targ.q2(s, o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
            # backup = ir + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
            backup = wr + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        s = data["sk"]
        o = data["obs"]
        pi, logp_pi = ac.pi(s, o)
        q1_pi = ac.q1(s, o, pi)
        q2_pi = ac.q2(s, o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up function for computing SAC pi loss
    def compute_loss_d(data):
        s = data["sk"]
        o = data["obs"]
        d = ac.di(o)

        # Cross entropy loss for the discriminator
        si = s.argmax(dim=1)
        loss_d = F.cross_entropy(d, si)

        # Useful info for logging
        d_prob = d.softmax(dim=1).detach()
        p_s = d_prob.gather(dim=1, index=si.unsqueeze(-1)).squeeze()
        d_info = dict(
            DiVals=d.detach().numpy(),
            DiProbS=p_s.numpy() # prob of actual skill
            # DiProbs=d_prob.detach().numpy(), # all probs
        )

        return loss_d, d_info
        
    # Set up optimizers for policy and q-function
    di_optimizer = Adam(ac.di.parameters(), lr=lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        
        # Next run on gradient descent step for the discriminator
        di_optimizer.zero_grad()
        loss_di, di_info = compute_loss_d(data)
        loss_di.backward()
        di_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        logger.store(LossDi=loss_di.item(), **di_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(s, o, deterministic=False):
        return ac.act(
            torch.as_tensor(s, dtype=torch.float32),
            torch.as_tensor(o, dtype=torch.float32), 
            deterministic
        )

    def test_agent():
        for j in range(num_test_episodes):
            sk, o, d, ep_ret, ep_iret, ep_wret, ep_len = g_sk(n_skill), test_env.reset(), False, 0, 0, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                dc = get_discriminator_confidence(sk, o)
                o, r, d, _ = test_env.step(get_action(sk, o, True))
                wr = compute_weighted_reward(dc, r)
                ep_wret += wr
                ep_iret += dc
                ep_ret += r
                ep_len += 1
            logger.store(
                TestEpIRet=ep_iret, TestEpWRet=ep_wret, TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    sk, o, ep_ret, ep_iret, ep_wret, ep_len = g_sk(n_skill), env.reset(), 0, 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(sk, o)
        else:
            a = env.action_space.sample()
            sk = g_sk(n_skill)

        # Step the env
        o2, r, d, _ = env.step(a)
        dc = get_discriminator_confidence(sk, o)
        wr = compute_weighted_reward(dc, r)
        ep_wret += wr
        ep_iret += dc
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(sk, o, a, r, dc, wr, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpIRet=ep_iret, EpWRet=ep_wret, EpRet=ep_ret, EpLen=ep_len)
            sk, o, ep_ret, ep_iret, ep_wret, ep_len = g_sk(n_skill), env.reset(), 0, 0, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", average_only=True)
            logger.log_tabular("EpIRet", average_only=True)
            logger.log_tabular("EpWRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", average_only=True)
            logger.log_tabular("TestEpIRet", average_only=True)
            logger.log_tabular("TestEpWRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("Q1Vals", with_min_and_max=True)
            logger.log_tabular("Q2Vals", with_min_and_max=True)
            logger.log_tabular("DiVals", average_only=True)
            logger.log_tabular("DiProbS", with_min_and_max=True)
            logger.log_tabular("LogPi", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("LossDi", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--n_skill", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="diayn")
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    diayn(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        n_skill=args.n_skill,
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
