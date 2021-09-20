"""
These 1D networks are suited for 1D inputs. 
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution, Normal

from ilpyt.nets.base_net import BaseNetwork, get_activation_layer


class DiscreteNetwork1D(BaseNetwork):
    def initialize(
        self,
        input_shape: Tuple,
        output_shape: int,
        num_layers: int = 2,
        num_hidden: int = 128,
        activation: str = 'relu',
        with_action_shape: int = 0,
    ) -> None:
        """
        1D network for discrete outputs.

        Parameters
        ----------
        input_shape: tuple
            shape of input to network
        output_shape: int
            shape of output of network
        num_layers: int, default=2
            number of linear layers to add to the network
        num_hidden: int, default=128
            hidden dimension of inner layers (number of filters)
        activation: str, default='relu'
            activation layer to add after hidden layers of network
        with_action_shape: int, default=0
            if specified, action will be incorporated into the net forward pass.
        """
        assert len(input_shape) == 1
        self.with_action_shape = with_action_shape
        input_shape = input_shape[0]
        if with_action_shape is not None:
            input_shape += with_action_shape
        activation_layer = get_activation_layer(activation)

        layers = [nn.Linear(input_shape, num_hidden)]  # type: List[nn.Module]
        for i in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(activation_layer)
        layers.append(nn.Linear(num_hidden, output_shape))
        self.layers = nn.Sequential(*layers)

        self.alpha = 4

    def get_action(self, x: torch.Tensor) -> Tuple[Distribution, torch.Tensor]:
        """
        Select an action by drawing from a distribution.

        Parameters
        ----------
        x: torch.Tensor
            input state tensor to network

        Returns
        -------
        torch.distributions.Distribution:
            distribution to sample actions from
        torch.Tensor:
            action tensor, sampled from distribution
        """
        logits = self.layers(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist, dist.sample()

    def forward(
        self, x: torch.Tensor, a: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass for network.

        Parameters
        ----------
        x: torch.Tensor
            input state tensor to network
        a: torch.Tensor, default=None
            optional; input action tensor

        Returns
        -------
        torch.Tensor:
            output tensor of forward pass
        """
        if a is not None and self.with_action_shape:
            a = F.one_hot(
                a.to(torch.int64), num_classes=self.with_action_shape
            )
            xa = torch.cat([x, a], dim=-1)
            return self.layers(xa)
        return self.layers(x)


class ContinuousNetwork1D(BaseNetwork):
    def initialize(
        self,
        input_shape: tuple,
        output_shape: int,
        num_layers: int = 2,
        num_hidden: int = 128,
        activation: str = 'relu',
        with_action_shape: int = 0,
    ) -> None:
        """
        1D network for continuous outputs.

        Parameters
        ----------
        input_shape: tuple
            shape of input to network
        output_shape: int
            shape of output of network
        num_layers: int, default=2
            number of linear layers to add to the network
        num_hidden: int, default=128
            hidden dimension of inner layers (number of filters)
        activation: str, default='relu'
            activation layer to add after hidden layers of network
        with_action_shape: int, default=0
            if specified, action will be incorporated into the net forward pass.
        """
        assert len(input_shape) == 1
        input_shape = input_shape[0]
        self.with_action_shape = with_action_shape
        if with_action_shape is not None:
            input_shape += with_action_shape
        activation_layer = get_activation_layer(activation)

        # added for sqil
        self.alpha = 4

        # Layers
        layers = [
            nn.Linear(input_shape, num_hidden)
        ]  # type: List[torch.nn.Module]
        for i in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(activation_layer)
        layers.append(nn.Linear(num_hidden, output_shape))
        self.layers = nn.Sequential(*layers)

        # Standard deviation
        log_std = 0.5 * torch.ones(output_shape)
        self.log_std = torch.nn.Parameter(log_std)

    def get_action(self, x: torch.Tensor) -> Tuple[Distribution, torch.Tensor]:
        """
        Select an action by drawing from a distribution.

        Parameters
        ----------
        x: torch.Tensor
            input state tensor to network

        Returns
        -------
        torch.distributions.Distribution:
            distribution to sample actions from
        torch.Tensor:
            action tensor, sampled from distribution
        """
        mu = self.layers(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        actions = dist.sample()
        return dist, actions

    def forward(
        self, x: torch.Tensor, a: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass for network.

        Parameters
        ----------
        x: torch.Tensor
            input state tensor to network
        a: torch.Tensor, default=None
            optional; input action tensor

        Returns
        -------
        torch.Tensor:
            output tensor of forward pass
        """
        if a is not None and self.with_action_shape:
            xa = torch.cat([x, a], dim=-1)
            return self.layers(xa)
        return self.layers(x)
