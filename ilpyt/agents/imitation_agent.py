"""
An implementation of a simple behavioral cloning (BC) agent, as in An Autonomous 
Land Vehicle in a Neural Network (ALVINN). The BC algorithm was described in 
the paper "An Autonomous Land Vehicle in a Neural Network" by Dean A. Pomerleau, and presented at NIPS 1988.

For more details, please refer to the paper: https://papers.nips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf
"""

from typing import Dict, Union

import numpy as np
import torch
from torch.optim import Adam

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.nets.base_net import BaseNetwork
from ilpyt.utils.agent_utils import *


class ImitationAgent(BaseAgent):
    def initialize(
        self, net: Union[BaseNetwork, None] = None, lr: float = 0.001
    ) -> None:
        """
        Initialization function for a simple BC agent.

        Parameters
        ----------
        net: BaseNetwork, default=None
            policy network
        lr: float, default=0.001
            learning rate

        Raises
        ------
        ValueError:
            if `net` is not specified
        """
        if net is None:
            raise ValueError(
                'Please provide input value for net. Currently set to None.'
            )
        self.net = net
        self.nets = {'net': self.net}
        self.opt = Adam(self.net.parameters(), lr=lr)

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
        _, actions = self.net.get_action(state)

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
            batch of transitions, with keys `states`, `actions`. Values 
            should be of size (num_steps, num_env, item_shape)

        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be 
            automatically recorded to TensorBoard
        """
        actions = batch['actions']
        if self.device == 'gpu':
            actions = actions.cuda()
        dist, _ = self.net.get_action(batch['states'])
        log_action_probs = dist.log_prob(actions)
        if len(log_action_probs.shape) > 1:
            log_action_probs = log_action_probs.sum(axis=-1)
        loss = -(log_action_probs.mean())

        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
        self.opt.step()

        loss_dict = {'loss/total': loss.item()}
        return loss_dict
