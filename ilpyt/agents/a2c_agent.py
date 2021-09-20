"""
An implementation of the agent from the Advantage Actor Critic (A2C) algorithm. 
This algorithm was described in the paper "Asynchronous Methods for Deep 
Reinforcement Learning" by Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi 
Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray 
Kavukcuoglu, and presented at ICML 2016. 

For more details, please refer to the paper: https://arxiv.org/abs/1602.01783
"""

from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.nets.base_net import BaseNetwork
from ilpyt.utils.agent_utils import compute_target, flatten_batch

__pdoc__ = {"__init__": False}


class A2CAgent(BaseAgent):
    def initialize(
        self,
        actor: Union[BaseNetwork, None] = None,
        critic: Union[BaseNetwork, None] = None,
        lr: float = 0.0001,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
    ) -> None:
        """
        Initialization function for the A2C agent.

        Parameters
        ----------
        actor: BaseNetwork, default=None
            actor network
        critic: BaseNetwork, default=None
            critic network
        lr: float, default=0.0001
            learning rate for training
        gamma: float, default=0.99
            discount factor used to compute targets
        entropy_coeff: float, default=0.01
            weight factor for entropy loss term

        Raises
        ------
        ValueError:
            if `actor` or `critic` are not specified
        """
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.lr = lr

        # Networks
        if actor is None or critic is None:
            raise ValueError(
                'Please provide BaseNetwork for actor and critic. Currently set to None.'
            )
        self.actor = actor
        self.critic = critic
        self.nets = {'actor': self.actor, 'critic': self.critic}

        self.opt_actor = Adam(self.actor.parameters(), self.lr)
        self.opt_critic = Adam(self.critic.parameters(), self.lr)

    @torch.no_grad()
    def reset(self) -> None:
        """
        Reset the network weights and optimizers.
        """
        # Reset actor weights
        for layers in self.actor.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # Reset critic weights
        for layers in self.critic.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # Reset optimizers
        self.opt_actor = Adam(self.actor.parameters(), self.lr)
        self.opt_critic = Adam(self.critic.parameters(), self.lr)

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Find best action for the given state according to the current policy.

        Parameters
        ----------
        state: torch.Tensor
            state tensor, of size (batch_size, state_shape)

        Returns
        -------
        np.ndarray:
            selected actions, of size (batch_size, action_shape)
        """
        _, actions = self.actor.get_action(state)

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
        # Compute values used for losses
        final_states = batch['next_states'][-1]
        value_final = self.critic(final_states).squeeze()
        targets = compute_target(
            value_final, batch['rewards'], 1 - batch['dones'], self.gamma
        ).reshape(-1)
        if self.device == 'gpu':
            targets = targets.cuda()

        batch = flatten_batch(batch)
        values = self.critic(batch['states']).squeeze()
        advantages = targets - values

        dist, _ = self.actor.get_action(batch['states'])
        actions = batch['actions']
        log_action_probs = dist.log_prob(actions)
        if len(log_action_probs.shape) > 1:
            log_action_probs = log_action_probs.sum(axis=-1)

        # Compute losses
        loss_action = -(log_action_probs * advantages.detach()).mean()
        loss_entropy = self.entropy_coeff * dist.entropy().mean()
        loss_actor = loss_action - loss_entropy
        loss_critic = F.smooth_l1_loss(values, targets)

        # Updates
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), 5
        )
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.opt_actor.step()
        self.opt_critic.step()

        # Return loss dictionary
        loss_dict = {
            'loss/actor': loss_actor.item(),
            'loss/critic': loss_critic.item(),
            'loss/total': loss_actor.item() + loss_critic.item(),
        }
        return loss_dict
