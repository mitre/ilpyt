"""
A generic trainer for reinforcement learning (RL) algorithms. Compatible with the 
Advantage Actor Critic (A2C), Deep Q-Network (DQN), and Proximal Policy 
Optimization (PPO) agents.
"""

import logging
import numpy as np

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.base_algo import BaseAlgorithm
from ilpyt.envs.vec_env import VecEnv
from ilpyt.runners.runner import Runner


class RL(BaseAlgorithm):
    def initialize(
        self,
        env: VecEnv,
        agent: BaseAgent,
        save_path: str = 'logs',
        load_path: str = '',
        use_gpu: bool = True,
    ):
        """
        Initialization function for an RL algorithm.

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
            If `agent` is not an instance of `A2CAgent`, `DQNAgent`, or `PPOAgent`.
        """
        self.env = env
        self.agent = agent
        self.use_gpu = use_gpu

        # Set up agent
        if (
            not isinstance(agent, A2CAgent)
            and not isinstance(agent, DQNAgent)
            and not isinstance(agent, PPOAgent)
        ):
            raise ValueError(
                'Behavioral cloning is only compatible with A2C, DQN, and PPO agents.'
            )
        if use_gpu:
            self.agent.to_gpu()
        if load_path:
            self.agent.load(load_path)

        # Set up runner
        self.runner = Runner(env, self.agent, use_gpu)

    def train(
        self,
        num_episodes: int = int(1e5),
        rollout_steps: int = 64
    ) -> None:
        """
        Train the agent in the specified environment.

        Parameters
        ----------
        num_episodes: int, default=1e5
            number of training episodes
        rollout_steps: int, 64
            number of rollout steps when generating a batch of experiences
        """
        self.best_reward = np.float('-inf')
        self.reward_tracker = self.best_reward * np.ones(self.env.num_envs)

        self.agent.set_train()
        step = 0
        ep_count = 0

        while ep_count < num_episodes:
            # Step agent and environment
            batch = self.runner.generate_batch(rollout_steps)
            step += rollout_steps

            # Update agent
            loss_dict = self.agent.update(batch)
            self.log(step, loss_dict)

            # Logging
            for ep_count, info_dict in batch['infos']:
                self.log(ep_count, info_dict)
                for (k, v) in info_dict.items():
                    if 'reward' in k:
                        agent_num = int(k.split('/')[1])
                        self.reward_tracker[agent_num] = v
            # should this be on this level?
            # Save agent
            reward = np.mean(self.reward_tracker)
            if reward > self.best_reward:
                self.agent.save(self.save_path, step)
                self.best_reward = reward
                logging.info(
                    "Save new best model at episode %i with reward %0.4f."
                    % (ep_count, reward)
                )

