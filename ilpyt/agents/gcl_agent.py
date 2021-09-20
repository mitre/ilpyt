"""
The agent from the Guided Cost Learning (GCL) algorithm. This algorithm was 
described in the paper "Guided Cost Learning: Deep Inverse Optimal Control via 
Policy Optimization" by PChelsea Finn, Sergey Levine, and Pieter Abbeel, and 
presented at ICML 2016.

For more details, please refer to the paper: https://arxiv.org/abs/1603.00448
"""

from typing import Dict, Union

import numpy as np
import torch
from torch.optim import Adam

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.nets.base_net import BaseNetwork


class GCLAgent(BaseAgent):
    def initialize(
        self,
        actor: Union[BaseNetwork, None] = None,
        cost: Union[BaseNetwork, None] = None,
        lr: float = 0.001,
        gamma: float = 0.99,
        clip_ratio: float = 0.1,
        entropy_coeff: float = 0.01,
        lcr_reg_cost: bool = False,
        mono_reg_cost: bool = False,
    ) -> None:
        """
        Initialization function for the GCL Agent.

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
        clip_ratio: float, default=0.1
            clipping parameter used in PPO loss function
        entropy_coeff: float, default=0.01
            entropy loss coefficient
        lcr_reg_cost: bool, default=False
            flag to add regularization term to demo and sample cost trajectories
        mono_reg_cost: bool, default=False
            flag to add mono regularization term to demo cost trajectory

        Raises
        ------
        ValueError:
            If `actor` or `critic` are not specified.
        """
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff

        # Networks
        if actor is None:
            raise ValueError(
                'Please provide input value for actor. Currently set to None.'
            )
        if cost is None:
            raise ValueError(
                'Please provide input value for critic. Currently set to None.'
            )
        self.actor = actor
        self.cost = cost
        self.nets = {'cost': self.cost, **self.actor.nets}

        self.opt_cost = Adam(self.cost.parameters(), lr)
        self.mono_reg_cost = mono_reg_cost
        self.lcr_reg_cost = lcr_reg_cost

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
        np.ndarray: selected actions
        """
        return self.actor.step(state)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update actor weights based on batch of experiences.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            batch of experiences, with values of size 
            (num_steps, num_env, item_shape)

        Returns
        -------
        Dict[str, float]: loss dictionary, with keys as tensorboard tags and 
                values as loss values to chart
        """
        # Rewards
        rollout_steps = batch['states'].shape[0]
        with torch.no_grad():
            rewards = []
            for i in range(rollout_steps):
                reward = -self.cost(batch['states'][i], batch['actions'][i])
                rewards.append(reward.squeeze())
            rewards = torch.stack(rewards)
        batch['rewards'] = rewards

        return self.actor.update(batch)

    def update_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update cost function weights based on batch of experiences.

        Parameters
        ----------
        states: torch.Tensor
            agent states, of size (batch_size, state_shape)
        actions: torch.Tensor
            agent actions, of size (batch_size, action_shape)
        expert_states: torch.Tensor
            expert states, of size (batch_size, state_shape)
        expert_actions: torch.Tensor
            expert actions, of size (batch_size, action_shape)

        Returns
        -------
        Dict[str, float]: loss dictionary, with keys as tensorboard tags and 
            values as loss values to log
        """
        sample_cost = self.cost(states, actions).squeeze()
        demo_cost = self.cost(expert_states, expert_actions).squeeze()

        with torch.no_grad():
            dist, _ = self.actor.actor.get_action(states)
            log_probs = dist.log_prob(actions)
            if len(log_probs.shape) > 1:  # continuous action space
                log_probs = log_probs.sum(axis=-1)
            probs = torch.exp(log_probs)

        loss_ioc = torch.mean(demo_cost) + torch.log(
            torch.mean(torch.exp(-sample_cost) / (probs + 1e-7))
        )

        return_log_dict = dict()
        # apply regularizers if you so wish
        # warning: computation time dramatically slower
        if self.lcr_reg_cost:
            demo_reg_lcr = self.apply_lcr_reg(demo_cost)
            sample_reg_lcr = self.apply_lcr_reg(sample_cost)
            loss_ioc += demo_reg_lcr + sample_reg_lcr
            return_log_dict["reg/demo_lcr"] = demo_reg_lcr.item()
            return_log_dict["reg/sample_lcr"] = sample_reg_lcr.item()

        if self.mono_reg_cost:
            demo_reg_mono = self.apply_mono_reg(demo_cost)
            loss_ioc += demo_reg_mono
            return_log_dict["reg/demo_mono"] = demo_reg_mono.item()

        self.opt_cost.zero_grad()
        loss_ioc.backward()
        torch.nn.utils.clip_grad_norm_(self.cost.parameters(), self.clip_ratio)
        self.opt_cost.step()

        return_log_dict['loss/sample_cost'] = torch.mean(sample_cost)
        return_log_dict['loss/demo_cost'] = torch.mean(demo_cost)
        return_log_dict['loss/ioc'] = loss_ioc.item()
        return return_log_dict

    def apply_lcr_reg(self, cost_traj_tensor: torch.Tensor):
        """
        Update cost function with local constant rate regularization term.

        Parameters
        ----------
        cost_traj_tensor: torch.Tensor
            cost tensor for the trajectory

        Returns
        --------
        torch.tensor: constant rate regularization tensor
        """
        cost_traj = cost_traj_tensor
        regularization_sum = torch.tensor(0.0)
        if self.device == 'gpu':
            regularization_sum = regularization_sum.cuda()

        for i in range(1, len(cost_traj) - 2):
            local_sum = (
                (cost_traj[i + 1] - cost_traj[i])
                - (cost_traj[i] - cost_traj[i - 1])
            ) ** 2
            regularization_sum += local_sum
        return regularization_sum

    def apply_mono_reg(self, cost_traj_tensor: torch.Tensor) -> torch.Tensor:
        """
        Update cost function with monotonic regularization term.

        Parameters
        ----------
        cost_traj_tensor: torch.Tensor
            cost tensor for the trajectory

        Returns
        -------
        torch.Tensor: monotonic regularization cost
        """
        cost_traj = cost_traj_tensor
        reg_sum = torch.tensor(0.0)
        zero_tensor = torch.tensor(0.0)
        if self.device == 'gpu':
            reg_sum = reg_sum.cuda()
            zero_tensor = zero_tensor.cuda()

        for i in range(1, len(cost_traj) - 1):
            local_max = (
                max(zero_tensor, cost_traj[i] - cost_traj[i - 1] - 1) ** 2
            )
            reg_sum += local_max
        return reg_sum

    def to_gpu(self) -> None:
        """
        Place agent nets on the GPU.
        """
        super(GCLAgent, self).to_gpu()
        self.actor.to_gpu()

    def to_cpu(self) -> None:
        """
        Place agent nets on the CPU.
        """
        super(GCLAgent, self).to_cpu()
        self.actor.to_cpu()

    def set_train(self) -> None:
        """
        Set agent nets to training mode.
        """
        super(GCLAgent, self).set_train()
        self.actor.set_train()

    def set_test(self) -> None:
        """
        Set agent nets to evaluation mode.
        """
        super(GCLAgent, self).set_test()
        self.actor.set_test()
