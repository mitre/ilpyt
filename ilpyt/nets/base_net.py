"""
`BaseNetwork` is the abstract class for a network. Networks parameterize 
important functions during learning - most often, the agent policy.

To create a custom network, simply extend `BaseNetwork`. The `BaseNetwork` API 
requires you to override the `initialize`, `get_action`, and `forward` methods.

- `initalize` sets `network` class variables, such as the network layers
- `get_action` draws from a torch distribution to perform an action
- `forward` computes a forward pass of the network
"""

from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch.distributions import Distribution


class BaseNetwork(torch.nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        **kwargs:
            arbitrary keyword arguments. Will be passed to the `initialize` and 
            `setup_experiment` functions
        """
        super(BaseNetwork, self).__init__()
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, input_shape: tuple, output_shape: int) -> None:
        """
        Perform network initialization. Build the network layers here. 
        Override this method when extending the `BaseNetwork` class.

        Parameters
        ----------
        input_shape: tuple
            shape of input to network
        output_shape: int
            shape of output of network
        """
        pass

    @abstractmethod
    def get_action(self, x: torch.Tensor) -> Tuple[Distribution, torch.Tensor]:
        """
        Some algorithms will require us to draw from a distribution to perform 
        an action. Override this method when extending the `BaseNetwork` class.

        Parameters
        ----------
        x: torch.Tensor
            input tensor to network

        Returns
        -------
        torch.distributions.Distribution:
            distribution to sample actions from
        torch.Tensor:
            action tensor, sampled from distribution
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network. Override this method when extending the 
        `BaseNetwork` class.

        Parameters
        ----------
        x: torch.Tensor
            input tensor to network

        Returns
        -------
        torch.Tensor:
            output of network
        """
        pass


def get_activation_layer(name: str) -> torch.nn.Module:
    """
    Get an activation layer with the given name.

    Parameters
    -----------
    name: str
        activation layer name, choose from 'relu' or 'tanh'

    Returns
    -------
    torch.nn.Module:
        activation layer

    Raises
    ------
    ValueError:
        if an unsupported activation layer is specified
    """
    if name == 'relu':
        return torch.nn.ReLU()
    elif name == 'tanh':
        return torch.nn.Tanh()
    else:
        raise ValueError('Unsupported activation layer chosen.')
