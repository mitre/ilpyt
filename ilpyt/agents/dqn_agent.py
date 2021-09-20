"""
An implementation of the agent from the Deep Q-Networks (DQN) algorithm. This 
algorithm was described in the paper "Human Level Control Through Deep 
Reinforcement Learning" by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, 
Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, 
Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, 
Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, 
Shane Legg, and Demis Hassabis, and published in Nature in February 2015. 

For more details, please refer to the paper: https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning
"""

import random
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.nets.base_net import BaseNetwork
from ilpyt.utils.agent_utils import flatten_batch, hard_update, soft_update
from ilpyt.utils.replay_memory import ReplayMemory


class DQNAgent(BaseAgent):
    def initialize(
        self,
        net: Union[BaseNetwork, None] = None,
        target_net: Union[BaseNetwork, None] = None,
        num_actions: int = -1,
        lr: float = 5e-5,
        replay_memory_size: int = int(1e4),
        epsilon_start: float = 0.95,
        epsilon_end: float = 0.01,
        epsilon_steps: int = int(1e5),
        tau: float = 0.01,
        gamma: float = 0.99,
        batch_size: int = 64,
        num_envs: int = 16,
    ) -> None:
        """
        Initialization function for the DQN Agent.

        Parameters
        ----------
        net: BaseNetwork, default=None
            deep q-network
        target_net: BaseNetwork, default=None
            target deep q-network
        replay_memory_size: int, default=1e4
            number of samples to store in replay memory
        num_actions: int, default=-1
            number of possible actions
        lr: float, default=5e-5
            learning rate
        epsilon_start: float, default=0.95
            probability between [0,1] of when to choose a random action
        epsilon_end: float, default=0.01
            probability between [0,1] of when to choose a random action
        epsilon_steps: int, default=1e5
            umber of steps to decrease from epsilon_start to epsilon_end
        tau: float, default=0.01
            soft update for target network [0, 1]
        gamma: float, default=0.99
            discount factor for estimating returns[0, 1]
        batch_size: int, default=64
            number of samples to take from replay_memory for a network update

        Raises
        ------
        ValueError:
            If `net` or `target_net` are not specified.
            If `num_actions` is not specified.
        """
        if num_actions == -1:
            raise ValueError(
                'Please provide valid input value for num_actions (positive integer). Currently set to -1.'
            )
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_envs = num_envs

        if net is None or target_net is None:
            raise ValueError(
                'Please provide input value for net and target_net. Currently set to None.'
            )
        self.net = net
        self.target_net = target_net
        hard_update(self.net, self.target_net)
        self.nets = {'net': self.net, 'target': self.target_net}

        self.lr = lr
        # Optimizer
        self.opt = Adam(self.net.parameters(), lr=self.lr)

        self.replay_memory_size = replay_memory_size
        # Replay memory
        self.memory = ReplayMemory(self.replay_memory_size)

        # Epsilon used for selecting actions
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_step = (epsilon_start - epsilon_end) / epsilon_steps
        self.epsilon_end = epsilon_end

    @torch.no_grad()
    def reset(self) -> None:
        """
        Reset the DQN agent network weights, replay memory, optimizers, and 
        epsilon.
        """
        # Reset net weights
        for layers in self.net.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        hard_update(self.net, self.target_net)
        self.memory = ReplayMemory(self.replay_memory_size)

        # Reset optimizers
        self.opt = Adam(self.net.parameters(), self.lr)

        # Reset epsilon
        self.epsilon = self.epsilon_start

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Find best action for the given state.

        Perform a random action with probability `self.epsilon`. Otherwise, 
        select the action which yields the maximum reward according to the
        current policy.

        Parameters
        ----------
        state: torch.Tensor
            state tensor, of size (batch_size, state_shape)

        Returns
        -------
        np.ndarray:
            selected actions, of size (batch_size, action_shape)
        """
        # Select epsilon
        self.epsilon = max(self.epsilon - self.epsilon_step, self.epsilon_end)

        # Perform random action with probability self.epsilon. Otherwise, select
        # the action which yields the maximum reward.
        if random.random() <= self.epsilon and self.mode == 'train':
            batch_size = state.shape[0]
            actions = np.random.choice(self.num_actions, batch_size)
        else:
            action_logits = self.net(state)
            actions = torch.argmax(action_logits, dim=-1)
            if self.device == 'gpu':
                actions = actions.cpu().numpy()
            else:
                actions = actions.numpy()
        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent policy based on batch of experiences.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            batch of transitions, with keys `states`, `actions`, `rewards`, 
            `dones`, and `next_states`. Values should be of size (num_steps, 
            num_env, item_shape)

        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be
            automatically recorded to TensorBoard
        """
        # Add to replay memory
        chunk_states = torch.chunk(batch['states'], self.num_envs, dim=1)
        chunk_next_states = torch.chunk(
            batch['next_states'], self.num_envs, dim=1
        )
        chunk_actions = torch.chunk(batch['actions'], self.num_envs, dim=1)
        chunk_rewards = torch.chunk(batch['rewards'], self.num_envs, dim=1)
        chunk_dones = torch.chunk(batch['dones'], self.num_envs, dim=1)

        for i in range(self.num_envs):
            rollout = {
                'states': chunk_states[i],
                'next_states': chunk_next_states[1],
                'actions': chunk_actions[i],
                'rewards': chunk_rewards[i],
                'dones': chunk_dones[i],
                'infos': [],
            }

            for ep_count, info_dict in batch['infos']:
                for (k, _) in info_dict.items():
                    if 'reward' in k and int(k.split('/')[1]) == i:
                        rollout['infos'].append([ep_count, info_dict])
            flattened = flatten_batch(rollout)
            self.memory.add(flattened)

        # Sample a batch
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return {}
        # Compute Q-values
        actions = batch['actions'].to(torch.int64).unsqueeze(1)
        action_logits = self.net(batch['states'])
        qs = action_logits.gather(-1, actions).squeeze()

        # Compute targets
        masks = 1 - batch['dones']
        target_action_logits = self.target_net(batch['next_states']).detach()
        target_max_action_logits = torch.max(
            target_action_logits, dim=-1
        ).values.detach()
        q_targets = (
            batch['rewards'] + self.gamma * masks * target_max_action_logits
        )

        # Compute loss
        loss = F.mse_loss(qs, q_targets)

        # Optimize model
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
        self.opt.step()

        # Update target
        soft_update(self.net, self.target_net, self.tau)

        # Return loss dictionary
        loss_dict = {'loss/total': loss.item()}
        return loss_dict
