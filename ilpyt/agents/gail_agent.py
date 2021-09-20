"""
An implementation of the agent from the Generative Adversarial Imitation 
Learning (GAIL) algorithm. This algorithm was described in the paper "Generative 
Adversarial Imitation Learning" by Jonathan Ho and Stefano Ermon, and presented 
at NIPS 2016.

For more details, please refer to the paper: https://arxiv.org/abs/1606.03476
"""

from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.nets.base_net import BaseNetwork
from ilpyt.utils.agent_utils import flatten_tensor


class GAILAgent(BaseAgent):
    def initialize(
        self,
        gen: Union[BaseAgent, None] = None,
        disc: Union[BaseNetwork, None] = None,
        lr: float = 1e-4,
    ) -> None:
        """
        Initialization function for the GAIL agent.

        Parameters
        ----------
        gen: BaseAgent, default=None
            actor (policy) network
        disc: BaseNetwork, default=None
            discriminator network
        lr: float, default=1e-4
            learning rate

        Raises
        ------
        ValueError:
            if `gen` is not specified, or is not an RL Agent (A2CAgent, 
            DQNAgent, or PPOAgent) or `disc` is not specified
        """
        # Networks
        if gen is None:
            raise ValueError(
                'Please provide input value for gen. Currently set to None.'
            )
        if disc is None:
            raise ValueError(
                'Please provide input value for disc. Currently set to None.'
            )
        if (
            not isinstance(gen, A2CAgent)
            and not isinstance(gen, DQNAgent)
            and not isinstance(gen, PPOAgent)
        ):
            raise ValueError(
                'GAILAgent.gen is only compatible with A2C, DQN, and PPO agents.'
            )
        self.gen = gen
        self.disc = disc
        self.nets = {'disc': self.disc, **self.gen.nets}
        self.opt_disc = Adam(self.disc.parameters(), lr)

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
        return self.gen.step(state)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent policy based on batch of experiences.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            batch of transitions, with keys `states`, `actions`, 
            `expert_states`, and `expert_actions`. Values should be of size 
            (num_steps, num_env, item_shape)

        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be 
            automatically recorded to TensorBoard
        """
        # Rewards
        rollout_steps = batch['states'].shape[0]
        with torch.no_grad():
            rewards = []
            for i in range(rollout_steps):
                logits = torch.sigmoid(
                    self.disc(batch['states'][i], batch['actions'][i])
                )
                reward = -torch.log(logits)
                rewards.append(reward.squeeze())
            rewards = torch.stack(rewards)

        # Update discriminator
        learner_logits = self.disc(
            flatten_tensor(batch['states']), flatten_tensor(batch['actions'])
        ).squeeze()
        expert_logits = self.disc(
            batch['expert_states'], batch['expert_actions']
        ).squeeze()
        loss_disc = F.binary_cross_entropy_with_logits(
            learner_logits, torch.ones_like(learner_logits)
        ) + F.binary_cross_entropy_with_logits(
            expert_logits, torch.zeros_like(expert_logits)
        )
        self.opt_disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.5)
        self.opt_disc.step()

        # Update generator
        batch['rewards'] = rewards
        loss_gen_dict = self.gen.update(batch)

        # Return loss dictionary
        loss_dict = {
            'loss/disc': loss_disc.item(),
            'loss/gen': loss_gen_dict['loss/total'],
            'loss/total': loss_disc.item() + loss_gen_dict['loss/total'],
        }
        return loss_dict

    def to_gpu(self) -> None:
        """
        Place agent nets on the GPU.
        """
        super(GAILAgent, self).to_gpu()
        self.gen.to_gpu()

    def to_cpu(self) -> None:
        """
        Place agent nets on the CPU.
        """
        super(GAILAgent, self).to_cpu()
        self.gen.to_cpu()

    def set_train(self) -> None:
        """
        Set agent nets to training mode.
        """
        super(GAILAgent, self).set_train()
        self.gen.set_train()

    def set_test(self) -> None:
        """
        Set agent nets to evaluation mode.
        """
        super(GAILAgent, self).set_test()
        self.gen.set_test()
