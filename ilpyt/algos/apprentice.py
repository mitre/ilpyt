"""
An implementation of the Apprenticeship Learning (AppL) algorithm. This 
algorithm was described in the paper "Apprenticeship Learning via Inverse 
Reinforcement Learning" by Pieter Abbeel and Andrew Ng, and presented at ICML 
2004.

For more details, please refer to the paper: https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf
"""

import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.base_algo import BaseAlgorithm
from ilpyt.envs.vec_env import VecEnv
from ilpyt.runners.runner import Runner


class Apprentice(BaseAlgorithm):
    def initialize(
        self,
        env: VecEnv,
        agent: BaseAgent,
        save_path: str = 'logs',
        load_path: str = '',
        use_gpu: bool = True,
    ) -> None:
        """
        Initialization function for the AppL algorithm.

        Parameters
        ----------
        env: VecEnv
            vectorized OpenAI Gym environment
        agent: BaseAgent
            agent for train and/or test. Must be an RL Agent
        save_path: str, default='logs'
            path to directory to save network weights
        load_path: str, default=''
            path to directory to load network weights. If not specified, network 
            weights will be randomly initialized.
        use_gpu: bool, default=True
            flag indicating whether or not to run operations on the GPU

        Raises
        ------
        ValueError:
            if `agent` is not an instance of `A2CAgent', 'PPOAgent', 'DQNAgent`
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
                'Apprenticeship Learning is only compatible with A2C, DQN, and PPO agents.'
            )
        if use_gpu:
            self.agent.to_gpu()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        if load_path:
            self.agent.load(load_path)

        # Set up runner
        self.runner = Runner(env, self.agent, use_gpu)

    def getFeatureExpectation(
        self,
        observations: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.9,
    ) -> torch.Tensor:
        """
        Calculate the feature expectation based off the given observations and 
        gamma value.

        Parameters
        ----------
        observations: torch.Tensor
            tensor of observations from env
        dones: torch.Tensor
            tensor of dones from the env 
            (indices where observation ends the episode)
        gamma: float, default = 0.9
            Discount value of observations

        Returns
        -------
        torch.Tensor: feature expectation tensor
        """

        done_indices = torch.where(dones)[0]
        if len(done_indices) == 0:
            done_indices = [len(dones) - 1]

        gamma_t = []
        for e in range(len(done_indices)):
            if e == 0:
                start = 0
            else:
                start = done_indices[e - 1] + 1
            end = done_indices[e] + 1
            gamma_t = gamma_t + [gamma ** x for x in range(end - start)]

        gamma_t = (
            torch.Tensor(gamma_t).unsqueeze(0).permute(1, 0).to(self.device)
        )
        sig_obs = torch.sigmoid(observations).to(self.device)

        discounted_observations = gamma_t * sig_obs
        fe = torch.sum(discounted_observations, dim=0) / len(done_indices)

        return fe

    def calculate_rewards(
        self, observations: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the new rewards using the given weights and observations.

        Parameters
        ----------
        observations: torch.Tensor
            tensor of observations from the current env
        weights: torch.Tensor
            reward weights for the observations

        Returns
        -------
        torch.Tensor: rewards vector
        """

        rewards = torch.matmul(torch.sigmoid(observations), weights)

        return rewards

    def initialize_step(self, gamma: float) -> Tuple[List, List, List, List]:
        """
        Initialize the buffers used in the AppL algorithm.
        Step one in the AppL algorithm for initialization.

        Parameters
        ----------
        gamma: float
            discount factor for feature expectation calculations

        Returns
        -------
        Tuple(List, List, List, List)
            feature_expectations, feature_expectations_bar, weights, margins
        """
        feature_expectations = []
        feature_expectations_bar = []
        weights = []
        margins = []

        # Get random policy feature expectation
        agent_batch = self.runner.generate_episodes(100)
        agent_fe = self.getFeatureExpectation(
            agent_batch.states, agent_batch.dones, gamma
        )
        feature_expectations.append(agent_fe)

        # Dummy weight value
        weights.append(torch.zeros(size=[self.env.observation_space.shape[0]]))

        # Dummy margin value
        margins.append(1)

        return feature_expectations, feature_expectations_bar, weights, margins

    def projection_method(
        self,
        feature_expectations: List,
        feature_expectations_bar: List,
        expert_fe: torch.Tensor,
        weights: List,
        margins: List,
        episode: int,
    ):
        """
        Perform the projection method from step 2 of the AppL algorithm.

        Parameters
        ----------
        feature_expectations: List
            feature_expectations list
        feature_expectations_bar: List
            feature_expectations_bar list
        expert_fe: torch.Tensor
            the feature expectations from the expert trajectories
        weights: List
            weights list
        margins: List
            margins list
        episode: int
            current episode
        """
        if episode == 1:
            feb = feature_expectations[0]
            feature_expectations_bar.append(feb)
            w = expert_fe - feature_expectations[0]
        else:
            A = feature_expectations_bar[episode - 2]
            B = feature_expectations[episode - 1] - A
            C = expert_fe - A

            feb = A + ((torch.dot(B, C) / torch.dot(B, B)) * B)
            feature_expectations_bar.append(feb)
            w = expert_fe - feature_expectations_bar[episode - 1]

        weights.append(w)

        t = torch.norm(expert_fe - feature_expectations_bar[episode - 1], p=2)
        margins.append(t)

    def train_rl_agent(
        self,
        batch_size: int,
        weights: torch.Tensor,
        num_train: int,
        episode: int,
    ):
        """
        Train the RL agent using the new reward function based on the 
        calculated weights.

        Parameters
        ----------
        batch_size: int
            batch size to train the agent with per training episode
        weights: torch.Tensor
            calculated weights for the reward function
        num_train: int
            number of episodes to train the RL agent for
        episode: int
            current episode
        """
        ep_count = 0
        step = 0
        while ep_count < num_train:
            batch_update = self.runner.generate_batch(batch_size)
            step += batch_size

            # Get calculated rewards
            new_rewards = self.calculate_rewards(
                batch_update['states'], weights[episode]
            )
            batch_update['rewards'] = new_rewards
            loss_dict = self.agent.update(batch_update)

            # Log loss metrics
            for (k, v) in loss_dict.items():
                temp = k.split('/')
                self.log(
                    step, {temp[0] + '_' + temp[1] + '/' + str(episode): v}
                )

            # Log training reward metrics
            for ep_count, info_dict in batch_update['infos']:
                for (k, v) in info_dict.items():
                    if 'reward' in k:
                        agent_num = int(k.split('/')[1])
                        self.log(
                            ep_count,
                            {
                                'reward_'
                                + str(episode)
                                + '/'
                                + str(agent_num): v
                            },
                        )

    def train(
        self,
        num_episodes: int = 100,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        batch_size: int = 64,
        num_train: int = 100,
        expert_demos: str = 'demos.pkl',
    ) -> None:
        """
        Train the AppL agent in the specified environment.

        Parameters
        ----------
        num_episodes: int, default=100
            number of training episodes for agent
        epsilon: float, default=0.1
            margin between expert and agent before early-stopping
        gamma: float,  default=0.9
            value used during feature expectation calculation
        batch_size: int, default=64
            batch size for steps to train
        num_train: int, default=100
            number of training episodes for the internal RL algo
        expert_demos: str, default='demos.pkl'
            path to expert demonstrations file, expects a *.pkl of a 
            `runner.Experiences` object
        """
        # Expert demonstrations
        with open(expert_demos, 'rb') as f:
            demos = pickle.load(f)  # runner.Experiences
            if self.use_gpu:
                demos.to_gpu()

        expert_observations = demos.states
        expert_dones = demos.dones
        expert_fe = self.getFeatureExpectation(
            expert_observations, expert_dones, gamma
        )

        # Step 1: Initialize
        logging.debug('Initializing variables...')
        (
            feature_expectations,
            feature_expectations_bar,
            weights,
            margins,
        ) = self.initialize_step(gamma)

        for episode in range(1, num_episodes + 1):
            # Step 2: Projection method
            self.projection_method(
                feature_expectations,
                feature_expectations_bar,
                expert_fe,
                weights,
                margins,
                episode,
            )

            logging.debug(
                'Episode %i, Margin: %0.5f' % (episode, margins[episode])
            )
            self.log(episode, {'overall/margin': margins[episode]})

            # Step 3: Termination
            if margins[episode] < epsilon:
                logging.info(
                    'Training complete at Episode %i with margin %.05f!'
                    % (episode, margins[episode])
                )
                break

            # Reset our agent since we did not terminate
            self.agent.reset()
            self.agent.set_train()
            self.runner.reset()

            # Step 4: Compute optimal policy
            logging.debug('Training RL agent...')
            self.train_rl_agent(batch_size, weights, num_train, episode)

            # Step 5: Get feature expectation for new policy
            agent_batch = self.runner.generate_test_episodes(100)
            batch_rewards = agent_batch.get_episode_rewards()
            mean_rewards = np.mean(batch_rewards)
            std_reward = np.std(batch_rewards)

            agent_fe = self.getFeatureExpectation(
                agent_batch.states, agent_batch.dones, gamma
            )

            # Log overall episode metrics
            self.log((episode - 1), {'overall/rewards': mean_rewards})
            logging.debug(
                'Episode %i, Mean Reward %.03f, STD %.03f'
                % (episode, mean_rewards, std_reward)
            )
            feature_expectations.append(agent_fe)

            self.agent.save(self.save_path, episode, keep=num_episodes)

        self.env.close()
