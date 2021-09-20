"""
An implementation of the Guided Cost Learning (GCL) algorithm. This algorithm 
was described in the paper "Guided Cost Learning: Deep Inverse Optimal Control 
via Policy Optimization" by PChelsea Finn, Sergey Levine, and Pieter Abbeel, and 
presented at ICML 2016.

For more details, please refer to the paper: https://arxiv.org/abs/1603.00448
"""

import copy
import logging
import pickle

import numpy as np
import torch

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.base_algo import BaseAlgorithm
from ilpyt.envs.vec_env import VecEnv
from ilpyt.runners.runner import Runner
from ilpyt.utils.agent_utils import flatten_batch


class GCL(BaseAlgorithm):
    def initialize(
        self,
        env: VecEnv,
        agent: BaseAgent,
        save_path: str = 'logs',
        load_path: str = '',
        use_gpu: bool = True,
    ):
        """
        Initialization function for the GCL algorithm.

        Parameters
        ----------
        env: VecEnv
            vectorized OpenAI Gym environment
        agent: BaseAgent
            agent for train and/or test
        save_path: str, default='logs'
            path to directory to save network weights
        load_path: str, default=''
            path to directory to load network weights. If not specified, network 
            weights will be randomly initialized
        use_gpu: bool, default=True
            flag indicating whether or not to run operations on the GPU

        Raises
        ------
        ValueError:
            if `agent` is not an instance of `A2CAgent`, `DQNAgent`, or `PPOAgent`
        """
        self.env = env
        self.agent = agent
        self.use_gpu = use_gpu

        # Set up agent
        if (
            not isinstance(agent.actor, A2CAgent)
            and not isinstance(agent.actor, DQNAgent)
            and not isinstance(agent.actor, PPOAgent)
        ):
            raise ValueError(
                'GCL is only compatible with A2C, DQN, and PPO actors.'
            )
        if use_gpu:
            self.agent.to_gpu()
        if load_path:
            self.agent.load(load_path)

        # Set up runner
        self.runner = Runner(env, self.agent, use_gpu)

    def train(
        self,
        num_episodes: int = int(1e4),
        num_reward_updates: int = 10,
        batch_size: int = 128,
        expert_demos: str = 'demos.pkl',
    ) -> None:
        """
        Train the agent within the specified environment.

        Parameters
        ----------
        num_episodes: int
            Number of training episodes
        num_reward_updates: int
            Number of times we update loss per rollout
        batch_size: int
            size of batches used to calculate loss and update network
        expert_demos: str
            path to expert demos pickle file
        """
        # Set train
        self.agent.set_train()
        self.best_loss = np.float('inf')
        self.best_reward = np.float('-inf')
        self.reward_tracker = self.best_reward * np.ones(self.env.num_envs)

        # Expert demonstrations
        with open(expert_demos, 'rb') as f:
            demos = pickle.load(f)  # runner.Experiences
            if self.use_gpu:
                demos.to_gpu()

        for i in range(num_episodes):
            # Generate samples
            batch = self.runner.generate_batch(64)
            flat_batch = flatten_batch(copy.deepcopy(batch))
            agent_batch_size = len(flat_batch['states'])
            expert_batch_size = len(demos.states)

            # Update cost function
            for j in range(num_reward_updates):
                selected_idxs = torch.randperm(expert_batch_size)[:batch_size]
                expert_states = demos.states[selected_idxs]
                expert_actions = demos.actions[selected_idxs]

                selected_idxs = torch.randperm(agent_batch_size)[:batch_size]
                states = flat_batch['states'][selected_idxs]
                actions = flat_batch['actions'][selected_idxs]

                states = torch.cat([states, expert_states], dim=0)
                actions = torch.cat([actions, expert_actions], dim=0)
                loss_cost_dict = self.agent.update_cost(
                    states, actions, expert_states, expert_actions
                )

            # Update policy
            loss_reward_dict = self.agent.update(batch)
            # Log
            self.log(i, loss_cost_dict)
            self.log(i, loss_reward_dict)

            # Save agent
            loss = loss_cost_dict['loss/ioc'] + loss_reward_dict['loss/total']

            # Logging
            for ep_count, info_dict in batch['infos']:
                self.log(ep_count, info_dict)
                for (k, v) in info_dict.items():
                    if 'reward' in k:
                        agent_num = int(k.split('/')[1])
                        self.reward_tracker[agent_num] = v

            mean_reward = np.mean(self.reward_tracker)

            self.log(i, {'values/mean_reward': mean_reward})

            # added in a check to make sure we aren't counting initial low loss
            if (loss < self.best_loss and i > 1000) or i % 500 == 0:
                self.agent.save(self.save_path, i)
                self.best_loss = loss
                logging.info(
                    "Save new best model at epoch %i with loss %0.4f."
                    % (i, loss)
                )

        self.env.close()
