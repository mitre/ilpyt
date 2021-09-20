import logging
from typing import Callable

import gym
import torch
from gym import spaces

import ilpyt.nets.net1d as net1d
import ilpyt.nets.net2d as net2d


import logging
from typing import Callable, Tuple, Union

import gym
import torch
from gym import spaces

import ilpyt.nets.base_net as BaseNetwork
import ilpyt.nets.net1d as net1d
import ilpyt.nets.net2d as net2d


def choose_net(
    env: gym.Env,
    input_shape: tuple = None,
    output_shape: int = None,
    activation: str = 'relu',
    with_action_shape: bool = False,
) -> torch.nn.Module:
    """
    From the available networks, choose the network class best 
    suited for the Gym environment according to the environment action space 
    and observation space.

    Parameters
    ----------
    env: gym.Env
        gym environment
    input_shape: tuple, default=None
        input dimensions for network. If not specified, set to the `env` 
        observation space
    output_shape: int, default=None
        output dimensions for network. If not specified, set to the 
        `env.num_actions`
    activation: str, default='relu'
        activation layer to use in the network, choose from [relu or tanh]
    with_action_shape: bool, default=False
        whether or not to include action in the network input

    Returns
    -------
    torch.nn.Module:
        available network that best suits given env
    """

    # Get action space type
    action_space = None
    if isinstance(env.action_space, spaces.Discrete):
        action_space = 'discrete'
    elif isinstance(env.action_space, spaces.Box):
        action_space = 'continuous'
    else:
        logging.error('Action space not supported.')

    # Get observation space type
    env_space = None
    if len(env.observation_space.shape) == 1:
        env_space = '1d'
    elif len(env.observation_space.shape) == 3:
        env_space = '3d'
    else:
        logging.error('Observation space not supported.')

    # Select network
    if action_space == 'discrete' and env_space == '1d':
        net = net1d.DiscreteNetwork1D  # type: Callable
    elif action_space == 'continuous' and env_space == '1d':
        net = net1d.ContinuousNetwork1D
    elif action_space == 'discrete' and env_space == '3d':
        net = net2d.DiscreteNetwork2D
    elif action_space == 'continuous' and env_space == '3d':
        net = net2d.ContinuousNetwork2D
    else:
        logging.error('Invalid combination of action and observation spaces.')

    if input_shape is None:
        input_shape = env.observation_shape
    if output_shape is None:
        output_shape = env.num_actions
    if with_action_shape:
        with_action_shape = env.num_actions

    return net(
        input_shape=input_shape,
        output_shape=output_shape,
        activation=activation,
        with_action_shape=int(with_action_shape),
    )
