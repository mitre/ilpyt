"""
An implementation of the Generative Adversarial Imitation Learning (GAIL) 
algorithm. This algorithm was described in the paper "Generative Adversarial 
Imitation Learning" by Jonathan Ho and Stefano Ermon, and presented at NIPS 2016.

For more details, please refer to the paper: https://arxiv.org/abs/1606.03476
"""

import logging
import pickle

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.gail_agent import GAILAgent
from ilpyt.algos.base_algo import BaseAlgorithm
from ilpyt.envs.vec_env import VecEnv
from ilpyt.runners.runner import Runner


class GAIL(BaseAlgorithm):
    def initialize(
        self,
        env: VecEnv,
        agent: BaseAgent,
        save_path: str = 'logs',
        load_path: str = '',
        use_gpu: bool = True,
    ) -> None:
        """
        Initialization function for the GAIL algorithm.

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
            If `agent` is not an instance of `GAILAgent`.
        """
        self.env = env
        self.agent = agent
        self.use_gpu = use_gpu

        # Set up agent
        if not isinstance(agent, GAILAgent):
            raise ValueError('GAIL is only compatible with GAILAgents.')
        if use_gpu:
            self.agent.to_gpu()
        if load_path:
            self.agent.load(load_path)

        # Set up runner
        self.runner = Runner(env, self.agent, use_gpu)

    def train(
        self,
        num_episodes: int = int(1e4),
        rollout_steps: int = 64,
        expert_demos: str = 'demos.pkl',
        save_interval: int = 1000,
    ) -> None:
        """
        Train the agent in the specified environment.

        Parameters
        ----------
        num_episodes: int, default=1e4
            number of training episodes
        rollout_steps: int, default=64
            rollout_steps
        expert_demos: str, default='demos.pkl'
            path to expert demonstrations file, expects a *.pkl of a 
            `runner.Experiences` object
        save_interval: int = 1000
            how often to save the model
        """
        num_save = int(num_episodes / save_interval)

        # Expert demonstrations
        with open(expert_demos, 'rb') as f:
            demos = pickle.load(f)  # runner.Experiences
            if self.use_gpu:
                demos.to_gpu()
        expert_dataset = TensorDataset(demos.states, demos.actions)
        expert_loader = DataLoader(
            expert_dataset,
            batch_size=self.env.num_envs * rollout_steps,
            shuffle=True,
        )
        expert_gen = iter(expert_loader)

        # Start training
        self.reward_tracker = np.float('-inf') * np.ones(self.env.num_envs)
        self.agent.set_train()
        step = 0
        ep_count = 0

        self.agent.save(self.save_path, 0, keep=num_save)
        logging.info("Save initial model at episode %i:%i." % (ep_count, step))

        while ep_count < num_episodes:
            # Step agent and environment
            batch = self.runner.generate_batch(rollout_steps)
            step += rollout_steps

            # Get expert rollouts
            try:
                expert_states, expert_actions = expert_gen.next()
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                expert_gen = iter(expert_loader)
                expert_states, expert_actions = expert_gen.next()
            batch['expert_states'] = expert_states
            batch['expert_actions'] = expert_actions

            # Update agent
            loss_dict = self.agent.update(batch)

            # Log
            self.log(step, loss_dict)
            for ep_count, info_dict in batch['infos']:
                self.log(ep_count, info_dict)
                for (k, v) in info_dict.items():
                    if 'reward' in k:
                        agent_num = int(k.split('/')[1])
                        self.reward_tracker[agent_num] = v

                # Save agent
                if ep_count % save_interval == 0:
                    self.agent.save(self.save_path, ep_count, keep=num_save)
                    logging.info(
                        "Save current model at episode %i:%i."
                        % (ep_count, step)
                    )

        self.agent.save(self.save_path, ep_count, keep=num_save)
        logging.info("Save final model at episode %i:%i." % (ep_count, step))
