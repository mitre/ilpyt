"""
An implementation of the agent from the Proximal Policy Optimization (PPO) 
algorithm. This algorithm was described in the paper "Proximal Policy
Optimization Algorithms" by John Schulman, Filip Wolski, Prafulla Dhariwal, 
Alec Radford, and Oleg Klimov, and published in 2017.

For more details, please refer to the paper: https://arxiv.org/abs/1707.06347
"""

from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.nets.base_net import BaseNetwork
from ilpyt.utils.agent_utils import compute_target, flatten_batch


class PPOAgent(BaseAgent):
    def initialize(
        self,
        actor: Union[BaseNetwork, None] = None,
        critic: Union[BaseNetwork, None] = None,
        lr: float = 0.001,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
    ) -> None:
        """
        Initialization function for the PPO agent.

        Parameters
        ----------
        actor: BaseNetwork, default=None
            actor network
        critic: BaseNetwork, default=None
            critic network
        lr: float, default=0.001
            learning rate
        gamma: float, default=0.99
            discount factor for calculating returns
        clip_ratio: float, default=0.2
            clipping parameter used in PPO loss function
        entropy_coeff: float, default=0.01
            entropy loss coefficient

        Raises
        ------
        ValueError:
            if `actor` or `critic` are not specified
        """
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.lr = lr

        # Networks
        if actor is None:
            raise ValueError(
                'Please provide input value for actor. Currently set to None.'
            )
        if critic is None:
            raise ValueError(
                'Please provide input value for critic. Currently set to None.'
            )
        self.actor = actor
        self.critic = critic
        self.nets = {'actor': self.actor, 'critic': self.critic}

        self.opt_actor = Adam(self.actor.parameters(), lr)
        self.opt_critic = Adam(self.critic.parameters(), lr / 2)

        # Track log probs
        # Not automatically recorded by the episode runner
        self.log_probs = []  # type: List[torch.Tensor]

    @torch.no_grad()
    def reset(self) -> None:
        """
        Reset the actor and critic layer weights and optimizers.
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
        Find best action for the given state.

        Parameters
        ----------
        state: torch.Tensor
            state tensor, of size (batch_size, state_shape)

        Returns
        -------
        np.ndarray:
            selected actions, of size (batch_size, action_shape)
        """
        dist, actions = self.actor.get_action(state)
        log_probs = dist.log_prob(actions)

        if len(log_probs.shape) > 1:  # continuous action space
            log_probs = log_probs.sum(axis=-1)
        self.log_probs.append(log_probs)

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
            `next_state`, and `dones`. Values should be of size 
            (num_steps, num_env, item_shape)

        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be 
            automatically recorded to TensorBoard
        """
        # Update critic
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
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        loss_critic = F.smooth_l1_loss(values, targets)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.5)
        self.opt_critic.step()

        # Update actor
        dist, _ = self.actor.get_action(batch['states'])
        log_action_probs = dist.log_prob(batch['actions'])
        if len(log_action_probs.shape) > 1:
            log_action_probs = log_action_probs.sum(axis=-1)
        if len(self.log_probs[0].shape) != 0:
            old_log_action_probs = torch.cat(self.log_probs)
        else:
            old_log_action_probs = torch.tensor(self.log_probs)
        if self.device == 'gpu':
            old_log_action_probs = old_log_action_probs.cuda()

        ratio = torch.exp(log_action_probs - old_log_action_probs.detach())
        clipped_advantages = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            * advantages.detach()
        )

        # Compute losses
        loss_entropy = self.entropy_coeff * dist.entropy().mean()
        loss_action = -(
            torch.min(ratio * advantages.detach(), clipped_advantages)
        ).mean()
        loss_actor = loss_action - loss_entropy

        # Updates
        self.opt_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.5)
        self.opt_actor.step()

        # Reset log_probs
        self.log_probs = []

        # Return loss dictionary
        loss_dict = {
            'loss/actor': loss_actor.item(),
            'loss/critic': loss_critic.item(),
            'loss/total': loss_actor.item() + loss_critic.item(),
        }
        return loss_dict
