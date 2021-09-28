"""
An implementation of the Dataset Aggregation (DAgger) algorithm. The DAgger 
algorithm was described in the paper "A Reduction of Imitation Learning and 
Structured Prediction to No-Regret Online Learning" by Stéphane Ross, 
Geoffrey J. Gordon, and J. Andrew Bagnell, and presented at AISTATS 2011.

For more details, please refer to the paper: https://arxiv.org/pdf/1011.0686.pdf
"""

import logging
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.base_algo import BaseAlgorithm
from ilpyt.envs.vec_env import VecEnv
from ilpyt.runners.runner import Runner


class DAgger(BaseAlgorithm):
    def initialize(
        self,
        env: VecEnv,
        agent: BaseAgent,
        save_path: str = 'logs',
        load_path: str = '',
        use_gpu: bool = True,
        max_mem: int = int(1e7),
    ) -> None:
        """
        Initialization function for the DAgger algorithm.

        Parameters
        ----------
        env: VecEnv
            vectorized OpenAI Gym environment
        agent: BaseAgent
            agent for train and/or test
        save_path: str, default='logs'
            path to directory to save network weights
        load_path: str, default=''
            path to directory to load network weights. If not specified, 
            network weights will be randomly initialized.
        use_gpu: bool, default=True
            flag indicating whether or not to run operations on the GPU.
        max_mem: int, default=1e7
            maximum number of transitions to store in memory

        Raises
        ------
        ValueError:
            If `agent` is not an instance of `ImitationAgent`
        """

        self.env = env
        self.agent = agent
        self.use_gpu = use_gpu
        self.max_mem = max_mem

        if not isinstance(agent, ImitationAgent):
            raise ValueError('DAgger is only compatible with ImitationAgents.')
        if use_gpu:
            self.agent.to_gpu()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        if load_path:
            self.agent.load(load_path)

        # Set up runner
        self.runner = Runner(env, self.agent, use_gpu)

    def update_dataset(
        self,
        dataset: dict,
        agent_batch_states: torch.Tensor,
        agent_actions: np.ndarray,
    ) -> TensorDataset:
        """
        Update given dataset with new state/action pairs.

        Parameters
        ----------
        dataset: dict
            dictionary of states and actions lists
        agent_batch_states: torch.Tensor
            batch states from the agent/expert
        agent_actions: torch.Tensor
            batch actions from the agent/expert

        Returns
        -------
        TensorDataset: dataset to be used during training
        """
        dataset['states'] = torch.cat(
            [dataset['states'], agent_batch_states], axis=0
        )
        dataset['actions'] = torch.cat(
            [dataset['actions'], torch.tensor(agent_actions).to(self.device)],
            axis=0,
        )

        # Cut dataset if larger than max memory
        if len(dataset['states']) > self.max_mem:
            dataset['states'] = dataset['states'][-self.max_mem :]
            dataset['actions'] = dataset['actions'][-self.max_mem :]

        tensor_dataset = TensorDataset(
            dataset['states'],
            dataset['actions'],
        )

        return tensor_dataset

    def check_best(
        self, batch_rewards: float, mean_loss: float, episode: int
    ) -> None:
        """
        Checks if the outputs from this batch beat the current bests. If the
        outputs include the current best loss, then save the model.

        Parameters
        ----------
        batch_rewards: float
            rewards from this batch
        mean_loss: float
            mean loss from this batch
        episode: int
            current episode
        """

        if batch_rewards > self.best_reward:
            self.best_reward = batch_rewards
            self.best_reward_loss = mean_loss

        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.best_loss_reward = batch_rewards
            self.agent.save(self.save_path, episode)
            logging.info(
                "Save new best model at episode %i with loss %0.4f and reward %0.3f."
                % (episode, mean_loss, batch_rewards)
            )

    def train(
        self,
        num_episodes: int = 100,
        batch_size: int = 20,
        num_epochs: int = 100,
        T_steps: int = 20,
        expert: Union[BaseAgent, None] = None,
    ) -> None:
        """
        Train the agent in the specified environment.

        Parameters
        ----------
        num_episodes: int, default=100
            number of training episodes for agent
        batch_size: int, default=20
            batch size for steps to train
        num_epochs: int, default=100
            number of epochs to train on aggregated dataset
        T_steps: int, default=20
             number of rollout steps for each update
        expert: BaseAgent, default=None
            interactive expert agent

        Raises
        ------
        ValueError:
            if `expert` is not specified
        """
        if expert is None:
            raise ValueError('Please provide expert. Currently set to None.')
        self.agent.set_train()

        expert_runner = Runner(self.env, expert, self.use_gpu)

        self.best_loss = np.float('inf')
        self.best_loss_reward = np.float('-inf')
        self.best_reward = np.float('-inf')
        self.best_reward_loss = np.float('inf')
        step = 0
        dataset = {
            'states': torch.Tensor([]).to(self.device),
            'actions': torch.Tensor([]).to(self.device),
        }

        for episode in range(num_episodes):
            decay = 0.5 ** (episode)

            # Generate trajectories
            with torch.no_grad():
                agent_probs = torch.distributions.Categorical(
                    torch.Tensor([decay, 1 - decay])
                )

                if agent_probs.sample() == 0:
                    # Get states sampled from the expert
                    agent_batch = expert_runner.generate_batch(T_steps)
                    model = 'Expert'
                else:
                    # Get states sampled from the agent
                    agent_batch = self.runner.generate_batch(T_steps)
                    model = 'Agent'

                agent_batch_states = agent_batch['states'].view(
                    -1, self.env.observation_shape[0]
                )
                agent_actions = expert.step(agent_batch_states)

            tensor_dataset = self.update_dataset(
                dataset, agent_batch_states, agent_actions
            )

            logging.info(
                "Episode %i:   %s Batch Size: %i    Total Dataset Size: %i"
                % (episode, model, len(agent_actions), len(tensor_dataset))
            )

            loader = DataLoader(
                tensor_dataset, batch_size=batch_size, shuffle=True
            )

            losses = []
            # Train learner on new dataset
            for _ in range(num_epochs):
                for (states, actions) in loader:
                    # Update agent
                    batch = {'states': states, 'actions': actions}
                    loss_dict = self.agent.update(batch)
                    losses.append(loss_dict['loss/total'])
                    step += 1

                    # Log
                    self.log(step, loss_dict)

            # Calculate mean loss and average reward for logging
            mean_loss = np.mean(losses)
            batch_rewards = np.mean(
                self.runner.generate_test_episodes(
                    100, start_seed=24
                ).get_episode_rewards()
            )

            # Log episode results
            rewards = {'rewards': batch_rewards}
            self.log(episode, rewards)
            logging.debug(
                'Episode %i, Mean Loss %0.4f, Reward %0.3f'
                % (episode, mean_loss, batch_rewards)
            )

            self.check_best(batch_rewards, mean_loss, episode)

        # Log the best model based on reward as well as the saved model 
        # (based on loss) for comparison
        logging.info(
            'Best Saved Model: Loss %0.4f, Reward %0.3f'
            % (self.best_loss, self.best_loss_reward)
        )
        logging.info(
            'Best Model Reward: Loss %0.4f, Reward %0.3f'
            % (self.best_reward_loss, self.best_reward)
        )
