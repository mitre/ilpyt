"""
These 2D networks are suited for 3D inputs (h,w,c). 
"""

from typing import Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution, Normal

from ilpyt.nets.base_net import BaseNetwork, get_activation_layer


class DiscreteNetwork2D(BaseNetwork):
    def initialize(
        self,
        input_shape: tuple,
        output_shape: int,
        activation: str = 'relu',
        with_action_shape: int = 0,
    ) -> None:
        """
        2D network for discrete outputs.

        Parameters
        ----------
        input_shape: tuple
            shape of input to network
        output_shape: int
            shape of output of network
        activation: str, default='relu'
            activation layer to add after hidden layers of network
        with_action_shape: int, default=0
            if specified, action will be incorporated into the net forward pass

        Raises
        ------
        AssertionError:
            if length of `input_shape` is not 3
        """
        assert len(input_shape) == 3

        activation_layer = get_activation_layer(activation)

        self.input_shape = (
            input_shape[2],
            input_shape[0],
            input_shape[1],
        )  # HWC to CHW
        in_channels = self.input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, 4),
            activation_layer,
            nn.Conv2d(16, 32, 4, 2),
            activation_layer,
            nn.Conv2d(32, 32, 4, 2),
            activation_layer,
        )

        self.with_action_shape = with_action_shape
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            activation_layer,
            nn.Linear(512, output_shape),
            nn.ReLU(),
        )

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
        x = x.permute(0, 3, 1, 2)
        x = x / 127.5 - 1
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        if a is not None and self.with_action_shape:
            a = F.one_hot(
                a.to(torch.int64), num_classes=self.with_action_shape
            )
            xa = torch.cat([x, a], dim=-1)
            return self.fc(xa)
        return self.fc(x)

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
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        actions = dist.sample()
        return dist, actions

    def feature_size(self):
        return (
            self.features(autograd.Variable(torch.zeros(1, *self.input_shape)))
            .view(1, -1)
            .size(1)
            + self.with_action_shape
        )


class ContinuousNetwork2D(BaseNetwork):
    def initialize(
        self,
        input_shape: tuple,
        output_shape: int,
        activation: str = 'relu',
        with_action_shape: int = 0,
    ) -> None:
        """
        2D network for discrete outputs.

        Parameters
        ----------
        input_shape: tuple
            shape of input to network
        output_shape: int
            shape of output of network
        activation: str, default='relu'
            activation layer to add after hidden layers of network
        with_action_shape: int, default=0
            if specified, action will be incorporated into the net forward pass

        Raises
        ------
        AssertionError:
            if length of `input_shape` is not 3
        """
        assert len(input_shape) == 3

        activation_layer = get_activation_layer(activation)

        self.input_shape = (
            input_shape[2],
            input_shape[0],
            input_shape[1],
        )  # HWC to CHW
        in_channels = self.input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            activation_layer,
            nn.Conv2d(32, 64, 4, 2),
            activation_layer,
            nn.Conv2d(64, 64, 3, 1),
            activation_layer,
        )

        self.with_action_shape = with_action_shape
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            activation_layer,
            nn.Linear(512, output_shape),
            nn.ReLU(),
        )

        # Standard deviation
        log_std = 0.5 * torch.ones(output_shape)
        self.log_std = torch.nn.Parameter(log_std)

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
        # Normalize
        x = x.permute(0, 3, 1, 2)
        x = x / 127.5 - 1

        # Forward
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        if a is not None and self.with_action_shape:
            xa = torch.cat([x, a], dim=-1)
            return self.fc(xa)
        return self.fc(x)

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
        mu = self.forward(x)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        actions = dist.sample()
        return dist, actions

    def feature_size(self):
        return (
            self.features(autograd.Variable(torch.zeros(1, *self.input_shape)))
            .view(1, -1)
            .size(1)
            + self.with_action_shape
        )
