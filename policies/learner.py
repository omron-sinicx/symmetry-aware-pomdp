# -*- coding: future_fstrings -*-
import os
import time

import math
import numpy as np
import random
import torch
from torch.nn import functional as F
import gym

# suppress this warning https://github.com/openai/gym/issues/1844
gym.logger.set_level(40)

import joblib

from .models import AGENT_CLASSES, AGENT_ARCHS

# Markov policy
from buffers.simple import SimpleBuffer

from buffers.seq_vanilla import SeqBuffer
from buffers.seq_per import SeqPerBuffer

from utils import helpers as utl
from utils.group import GroupHelper
from torchkit import pytorch_utils as ptu
from utils import logger
from utils.helpers import EpisodeLogger

import wandb


class Learner:
    def __init__(self, env_args, train_args, eval_args,
                 policy_args, seed, replay, time_limit,
                 explore_deterministic,
                 prefix, ckpt_dir, replay_policy, cfg_file, **kwargs):
        self.seed = seed
        self.group_prefix = prefix

        self.replay = replay

        self.time_limit = time_limit

        self.explore_deterministic = explore_deterministic

        # TODO:
        self.per_expert_eps = 1.0
        self.per_eps = 1e-6

        if train_args['use_expert_rollouts'] > 0:
            self.demonstration_dir = f"demonstrations/{env_args['env_name']}/"
        
        if 'group_name' in policy_args:
            group_name = policy_args["group_name"]
        else:
            group_name = "none"
        ckpt_filename = f"{env_args['env_name'][:-3]}"      \
                        + f"_{policy_args['algo_name']}"    \
                        + f"_{policy_args['actor_type']}"   \
                        + f"_{policy_args['critic_type']}"  \
                        + f"_g{group_name}" \
                        + f"_e{train_args['use_expert_rollouts']}"
        
        ckpt_filename += f"_s{self.seed}.pt"

        if ckpt_dir is not None:
            self.checkpoint_dir = os.path.join(ckpt_dir, ckpt_filename)
        else:
            self.checkpoint_dir = ckpt_filename

        self.checkpoint_dir_end = self.checkpoint_dir + '_end'

        if os.path.exists(self.checkpoint_dir):
            self.chkpt_dict = joblib.load(self.checkpoint_dir)
            logger.log("Load checkpoint file done")
            self.set_random_state()
        else:
            # this file exists, then this run has ended
            if os.path.exists(self.checkpoint_dir_end):
                logger.log("End file found, exit")
                exit(0)
            logger.log("Checkpoint file not found")
            self.chkpt_dict = None

        if not replay_policy:
            self.init_wandb(cfg_file, **env_args, **policy_args, **train_args)

        self.init_env(**env_args)

        self.init_agent(**policy_args, **env_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)


    def set_random_state(self):
        random.setstate(self.chkpt_dict["random_rng_state"])
        np.random.set_state(self.chkpt_dict["numpy_rng_state"])
        torch.set_rng_state(self.chkpt_dict["torch_rng_state"])
        torch.cuda.set_rng_state(self.chkpt_dict["torch_cuda_rng_state"])
        logger.log("Load random state checkpoint done")

    def init_env(
        self,
        env_type,
        env_name,
        num_eval_tasks=None,
        **kwargs
    ):

        # initialize environment
        assert env_type in [
            "pomdp",
        ]
        self.env_type = env_type

        if self.env_type in [
            "pomdp",
        ]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp

            assert num_eval_tasks > 0
            try:
                self.train_env = gym.make(env_name, rendering=self.replay)
            except TypeError:
                self.train_env = gym.make(env_name)  # gym env does not have rendering keyword
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = gym.make(env_name)
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        if len(self.train_env.observation_space.shape) == 3:
            self.obs_dim = self.train_env.observation_space.shape
        else:
            self.obs_dim = self.train_env.observation_space.shape[0]  # include 1-dim done

        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
        actor_type,
        critic_type,
        image_encoder=None,
        group_name=None,
        env_name=None,
        **kwargs
    ):
        # initialize agent

        if actor_type == 'mlp':
            agent_class = AGENT_CLASSES["Policy_MLP"]
        else:
            agent_class = AGENT_CLASSES["Policy_Separate_RNN"]

        self.agent_arch = agent_class.ARCH

        if self.chkpt_dict is None:
            logger.log(agent_class, self.agent_arch)

        if group_name is not None:
            group_helper = GroupHelper(group_name, env_name, actor_type, critic_type)
        else:
            group_helper = None

        if image_encoder is not None:
            raise NotImplementedError
        else:
            actor_image_encoder_fn = lambda: None
            critic_image_encoder_fn = lambda: None

        self.agent = agent_class(
            actor_type=actor_type,
            critic_type=critic_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            actor_image_encoder_fn=actor_image_encoder_fn,
            critic_image_encoder_fn=critic_image_encoder_fn,
            group_helper=group_helper,
            **kwargs,
        ).to(ptu.device)

        if self.chkpt_dict is not None:
            self.agent.restore_state_dict(self.chkpt_dict["agent_dict"])
            logger.log("Load agent checkpoint done")

        if self.chkpt_dict is None:
            logger.log(self.agent)

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters,
        num_init_rollouts_pool,
        use_expert_rollouts,
        num_rollouts_per_iter,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        **kwargs
    ):

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            # no rotational augmentation
            if buffer_type == SeqBuffer.buffer_type:
                buffer_class = SeqBuffer

            # prioritized replay buffer
            elif buffer_type == SeqPerBuffer.buffer_type:
                buffer_class = SeqPerBuffer

            else:
                raise ValueError("Buffer type not supported.")

            if self.chkpt_dict is None:
                logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
            )

        # load buffer from checkpoint
        if self.chkpt_dict is not None:
            self.policy_storage.load_from_state_dict(
                    self.chkpt_dict["buffer_dict"])
            logger.log("Load buffer checkpoint done")

        self.batch_size = batch_size
        self.num_iters = num_iters

        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.use_expert_rollouts = use_expert_rollouts
        self.num_rollouts_per_iter = num_rollouts_per_iter

        total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
        self.n_env_steps_total = self.max_trajectory_len * total_rollouts
        logger.log(
            "*** total rollouts",
            total_rollouts,
            "total env steps",
            self.n_env_steps_total,
        )

    def init_eval(
        self,
        log_interval,
        save_interval,
        log_tensorboard,
        num_episodes_per_task=1,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.log_tensorboard = log_tensorboard
        self.eval_num_episodes_per_task = num_episodes_per_task

    def init_wandb(
        self,
        cfg_file,
        env_name,
        algo_name,
        actor_type,
        critic_type,
        group_name=None,
        **kwargs,
    ):
        # Same project for state env
        if 'State-' in env_name:
            filtered_env_name = env_name.replace('State-', '')
        else:
            filtered_env_name = env_name

        project_name = f"Omron_{filtered_env_name[:-3]}"

        if group_name is not None:
            group_info = group_name
        else:
            group_info = ""

        group = f"{algo_name}_{actor_type}_{critic_type}_" + \
                f"g{group_info}"

        if self.group_prefix is not None:
            group = f"{self.group_prefix}_{group}"

        if 'State-' in env_name:
            group = f"state_{group}"

        wandb_args = {}
        checkpoint_needed = (actor_type == "equi") and (critic_type == "equi")
        if self.chkpt_dict is not None and checkpoint_needed:
            wandb_args = {"resume": "allow",
                          "id": self.chkpt_dict["wandb_id"]}
        else:
            wandb_args = {"resume": None}
        
        wandb.login(key='716c327163fa74839eebff06e5d42277ef7c1896')
        wandb.init(project=project_name,
                   settings=wandb.Settings(_disable_stats=True),
                   entity='hainh22',
                   group=group,
                   name=f"s{self.seed}",
                   **wandb_args)
        wandb.save(cfg_file)

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._last_time = 0

        self._best_per = 0.0

        if self.chkpt_dict is not None:
            self._n_env_steps_total = self.chkpt_dict["_n_env_steps_total"]
            self._n_rollouts_total = self.chkpt_dict["_n_rollouts_total"]
            self._n_rl_update_steps_total = self.chkpt_dict["_n_rl_update_steps_total"]
            self._n_env_steps_total_last = self.chkpt_dict["_n_env_steps_total_last"]
            self._last_time = self.chkpt_dict["_last_time"]
            self._best_per = self.chkpt_dict["_best_per"]
            logger.log("Load training statistic done")
            # final load so save some memory
            self.chkpt_dict = {}

        self._start_time = time.time()
        self._start_time_last = time.time()

    def train(self, initial_model):
        """
        training loop
        """

        self._start_training()

        if initial_model is not None:
            self.load_model(initial_model)
            self.agent.reset_optimizers()

        if self.use_expert_rollouts > 0 and self.chkpt_dict is None:
            logger.log("Loading expert rollouts...")
            self.load_expert_rollouts()
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float) and initial_model is None:
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter),
                )
                self.log_train_stats(train_stats)
            
        if self.num_init_rollouts_pool > 0 and self.chkpt_dict is None:
            logger.log("Collecting initial pool of random data...")
            env_steps = self.collect_rollouts(
                num_rollouts=self.num_init_rollouts_pool,
                random_actions=True,
            )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(env_steps * self.num_updates_per_iter)
                )
                self.log_train_stats(train_stats)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            env_steps = self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter,
                                              deterministic=self.explore_deterministic)
            logger.log("env steps", self._n_env_steps_total)

            # Save checkpoint
            if (time.time() - self._start_time)/3600.0 > self.time_limit:
                self.checkpoint()
                exit(0)

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps)),
            )  # NOTE: ceil to make sure at least 1 step
            self.log_train_stats(train_stats)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters
                perf = self.log()

                # save best model
                if (
                    self.save_interval > 0 and
                    self._n_env_steps_total >= 0.75 * self.n_env_steps_total
                ):
                    if perf > self._best_per:
                        logger.log(f"Replace {self._best_per} w/ {perf} model")
                        self._best_per = perf
                        self.save_model(current_num_iters, perf,
                                        wandb_save=True)

                # save model according to a frequency
                if (
                    self.save_interval > 0
                    and current_num_iters % self.save_interval == 0
                ):
                    # save models in later training stage
                    self.save_model(current_num_iters, perf, wandb_save=True)
        self.save_model(current_num_iters, perf, wandb_save=True)

        if os.path.exists(self.checkpoint_dir):
            # remove checkpoint file to save space
            os.system(f"rm {self.checkpoint_dir}")
            logger.log("Remove checkpoint file")

            # create a file to signify that this run has ended
            joblib.dump(
                    {
                        "_n_env_steps_total": self._n_env_steps_total
                    },
                    self.checkpoint_dir_end
            )

    def checkpoint(self):
        logger.log(f"Saving checkpoint {self.checkpoint_dir}...")

        # save replay buffer data
        buffer_dict = self.policy_storage.get_state_dict()

        self._last_time += (time.time() - self._start_time) / 3600.0

        joblib.dump(
            {
                "buffer_dict": buffer_dict,

                "agent_dict": self.agent.get_state_dict(),

                "random_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state(),

                "_n_env_steps_total": self._n_env_steps_total,
                "_n_rollouts_total": self._n_rollouts_total,
                "_n_rl_update_steps_total": self._n_rl_update_steps_total,
                "_n_env_steps_total_last": self._n_env_steps_total_last,

                "wandb_id": wandb.run.id,

                "_last_time": self._last_time,
                "_best_per": self._best_per
            },
            self.checkpoint_dir,
        )
        logger.log("Checkpointing done and exit")

    def replay_policy(self, policy_dir):
        """
        replay a policy
        """
        self._start_training()

        self.load_model(policy_dir)

        last_eval_num_iters = 0
        while self._n_env_steps_total < self.n_env_steps_total:
            # collect data from num_rollouts_per_iter train tasks:
            self.collect_rollouts(num_rollouts=self.num_rollouts_per_iter, deterministic=True)
            logger.log("env steps", self._n_env_steps_total)

            # evaluate and log
            current_num_iters = self._n_env_steps_total // (
                self.num_rollouts_per_iter * self.max_trajectory_len
            )
            if (
                current_num_iters != last_eval_num_iters
                and current_num_iters % self.log_interval == 0
            ):
                last_eval_num_iters = current_num_iters

    @torch.no_grad()
    def load_expert_rollouts(self):
        """load expert rollouts from file
        """
        directory = self.demonstration_dir
        # Get the list of files in the directory
        files = os.listdir(directory)

        # Filter and count the number of npz files
        npz_files = [file for file in files if file.endswith('.npz')]

        before_env_steps = self._n_env_steps_total

        for file in npz_files:
            file_path = os.path.join(directory, file)

            # print("Processing:", file_path)

            data = np.load(file_path)
            log_obs = data['obs']
            log_action = data['action']
            log_next_obs = data['next_obs']
            log_reward = data['reward']
            log_done = data['done']

            steps = 0

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            for i in range(len(log_action)):
                action = ptu.FloatTensor(log_action[i]).reshape(1, -1)  # (1, A) for continuous action, (1) for discrete action

                # load reward and next obs (B=1, dim)
                obs = log_obs[i]
                next_obs = log_next_obs[i]
                reward = log_reward[i]
                done = log_done[i]  # done will be True only at the final timestep

                # convert to tensors
                obs = ptu.from_numpy(obs).view(-1, *obs.shape)
                next_obs = ptu.from_numpy(next_obs).view(-1, *next_obs.shape)
                reward = ptu.FloatTensor([reward]).view(-1, 1)
                done = ptu.from_numpy(np.array(done, dtype=int)).view(-1, 1)

                # update statistics
                steps += 1

                # term ignore time-out scenarios, but record early stopping
                term = log_reward[i] > 0

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )

                    if log_done[i]:
                        print(
                            f"expert steps: {steps} term: {log_done[i]} ret: {log_reward[i]}"
                        )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                _reward = torch.cat(rew_list, dim=0).sum().item()
                assert _reward > 0
                valid_episode = False
                if len(obs_list) <= self.max_trajectory_len:  # can add stricter condition here
                    valid_episode = self.policy_storage.add_episode(
                        observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                        actions=ptu.get_numpy(act_buffer),  # (L, dim)
                        rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                        terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                        next_observations=ptu.get_numpy(
                            torch.cat(next_obs_list, dim=0)
                        ),  # (L, dim)
                        expert_masks=np.ones_like(term_list).reshape(-1, 1),  # (L, 1)
                    )

                    print(
                        f"expert steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                    )
            else:
                valid_episode = True
            if valid_episode:
                self._n_env_steps_total += steps
                self._n_rollouts_total += 1

        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False, deterministic=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset())  # reset

            obs = obs.reshape(1, *obs.shape)

            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            obs=obs,
                            deterministic=deterministic,
                        )
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=deterministic)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                # term ignore time-out scenarios, but record early stopping
                term = (
                    False
                    if "TimeLimit.truncated" in info
                    or steps >= self.max_trajectory_len
                    else done_rollout
                )

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            add_episode_success = False

            if self.agent_arch == AGENT_ARCHS.Markov:
                add_episode_success = True
                if done_rollout:
                    print(
                        f"steps: {steps} term: {term} ret: {reward.cpu().item()}"
                    )

            if self.agent_arch in [AGENT_ARCHS.Memory]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)
                add_episode_success = self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                    expert_masks=np.zeros_like(term_list).reshape(-1, 1),  # (L, 1)
                )

                if add_episode_success:
                    print(
                        f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                    )

            if add_episode_success:
                self._n_env_steps_total += steps
                self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def save_rollouts(self, policy_dir, num_rollouts=100):
        """collect num_rollouts of trajectories in task and save a file
        """
        self.load_model(policy_dir)

        cnt_rollouts = 0

        while True:
            if cnt_rollouts == num_rollouts:
                break

            data_reader = EpisodeLogger(str(cnt_rollouts))
            episode_data = []

            obs = ptu.from_numpy(self.train_env.reset())  # reset
            state = self.train_env.get_state()

            obs = obs.reshape(1, *obs.shape)

            done_rollout = False

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                # policy takes hidden state as input for memory-based actor,
                # while takes obs for markov actor
                if self.agent_arch == AGENT_ARCHS.Memory:
                    (action, _, _, _), internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        obs=obs,
                        deterministic=True,
                    )
                else:
                    action, _, _, _ = self.agent.act(obs, deterministic=True)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )
                next_state = self.train_env.get_state()

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # add data to files
                episode_data.append((ptu.get_numpy(obs)[0],
                                     ptu.get_numpy(action)[0],
                                     ptu.get_numpy(next_obs)[0],
                                     ptu.get_numpy(reward)[0, 0],
                                     done_rollout,
                                     state,
                                     next_state))

                # only save successful episodes
                if reward > 0.0:
                    data_reader.log_episode(episode_data)
                    print(f"Saved rollout: {cnt_rollouts}")
                    cnt_rollouts += 1
                    break

                # set: obs <- next_obs, state <- next_state
                obs = next_obs.clone()
                state = next_state.copy()

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
            episode_indices, weights = None, None
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch, episode_indices, weights = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch), episode_indices, weights

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch, episode_indices, weights = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch, weights)

            # update priorities if using prioritized replay buffer
            if isinstance(self.policy_storage, SeqPerBuffer):
                new_priorities = (ptu.get_numpy(rl_losses['avg_abs_td_errors'])
                                  +
                                  ptu.get_numpy(batch["exp_msk"][0, :, :]) *  # bonus for expert episodes
                                  self.per_expert_eps
                                  +
                                  self.per_eps)
                self.policy_storage.update_priorities(episode_indices,
                                                      new_priorities)

            for k, v in rl_losses.items():
                if k != "avg_abs_td_errors":
                    if update == 0:  # first iterate - create list
                        rl_losses_agg[k] = [v]
                    else:  # append values
                        try:
                            rl_losses_agg[k].append(v)
                        except KeyError:
                            pass
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):

        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        for task_idx, task in enumerate(tasks):
            step = 0

            obs = ptu.from_numpy(self.eval_env.reset())  # reset

            obs = obs.reshape(1, *obs.shape)

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, internal_state = self.agent.get_initial_info()

            for _ in range(self.max_trajectory_len):
                if self.agent_arch == AGENT_ARCHS.Memory:
                    (action, _, _, _), internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        obs=obs,
                        deterministic=deterministic,
                    )
                else:
                    action, _, _, _ = self.agent.act(
                        obs, deterministic=deterministic
                    )

                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )

                step += 1
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # set: obs <- next_obs
                obs = next_obs.clone()

                if "success" in info and info["success"] is True:
                    assert done_rollout == True
                    success_rate[task_idx] = 1.0

                if done_rollout:
                    # for all env types, same
                    break
            total_steps[task_idx] = step
        return success_rate, total_steps

    def log_train_stats(self, train_stats):
        ## log losses
        for k, v in train_stats.items():
            wandb.log({"rl_loss/" + k: v}, step=self._n_env_steps_total)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                wandb.log({"rl_loss/" + k: v}, step=self._n_env_steps_total)

    def log(self):
        # --- evaluation ----
        success_rate_eval, total_steps_eval = self.evaluate(
            self.eval_tasks
        )

        logger.log("eval/success_rate", np.mean(success_rate_eval))

        wandb.log(
                 {
                    'env_steps': self._n_env_steps_total,
                    'rollouts': self._n_rollouts_total,
                    'rl_steps': self._n_rl_update_steps_total,
                    'metrics/success_rate_eval': np.mean(success_rate_eval),
                    'metrics/total_steps_eval': np.mean(total_steps_eval),
                    'time_cost': (time.time() - self._start_time)/3600 + self._last_time,
                    'fps': (self._n_env_steps_total - self._n_env_steps_total_last)
                    / (time.time() - self._start_time_last)
                 },
                 step=self._n_env_steps_total
                 )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        return np.mean(success_rate_eval)

    def save_model(self, iter, perf, wandb_save=False, save_pretrained_expert=False):
        if save_pretrained_expert:
            fname = "agent_expert.pt"
        else:
            fname = f"agent_{iter}_perf{perf:.3f}.pt"
        save_path = os.path.join(
            logger.get_dir(), "save", fname
        )
        torch.save(self.agent.state_dict(), save_path)

        if wandb_save:
            logger.log(f"Save file {fname} to wandb")
            wandb.save(save_path)

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path,
                                   map_location=ptu.device))
        print("load successfully from", ckpt_path)
