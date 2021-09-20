"""
`BaseAgent` is the abstract class for an agent. An agent's role during learning 
is to coordinate the policy learning and execution. Here, the policy refers to 
a function (in this case, a deep neural network), which maps states to actions.

To create a custom agent, simply extend `BaseAgent`. The `BaseAgent` API 
requires you to override the `initialize`, `step` and `update` methods.

- `initalize` sets `agent` class variables, such as the agent optimizers and 
policy functions.
- `step` ingests a state and outputs an action, 
- `update` ingests a batch of transitions to update the policy weights. 
"""
import logging
import os
from abc import ABCMeta, abstractmethod
from glob import glob
from typing import Dict

import numpy as np
import torch


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        By default, the agent will be in `train` mode and be configured to use 
        the `cpu` for `step` and `update` calls.

        Parameters
        ----------
        **kwargs:
            arbitrary keyword arguments that will be passed to the `initialize` function
        """
        self.mode = 'train'
        self.device = 'cpu'
        self.nets = {}  # Instantiate this in the initialize function.
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self) -> None:
        """
        Perform agent initialization. This could include setting class variables 
        for agent optimizers, agent networks, and other important values. 
        Override this method when extending the `BaseAgent` class.

        Make sure to add all the agent networks to a `self.nets` class variable 
        in a dictionary format, where the key is a string containing the network 
        name and the value is the network instance (an extension of the 
        `nets.base_net.BaseNetwork` class). The `self.nets` class variable will 
        become important when we `save` and `load` network weights.

        Parameters
        ----------
        **kwargs:
            arbitrary keyword arguments
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Have the agent take an action in the environment. Override this method 
        when extending the `BaseAgent` class, and be sure to apply the 
        `torch.no_grad()` decorator.

        Parameters
        ----------
        state: torch.Tensor
            batch of environment state vectors, which will be of shape 
            (batch_size, state_shape)

        Returns
        -------
        np.ndarray:
            batch of selected actions, which will be of shape 
            (batch_size, action_shape)
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent policy according to a batch of transitions. Override 
        this method when extending the `BaseAgent` class.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            batch of transitions. Keys for the dictionary at a minimum will 
            include `states` and `actions`, but may also include `rewards` and 
            other values needed for the policy update

        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be 
            automatically recorded to TensorBoard (see 
            `algos.BaseAlgorithm.log`) function for more details
        """
        pass

    def save(self, save_path: str, step: int, keep: int = 3) -> None:
        """
        Save the agent network(s) weights, according to `self.nets`. All 
        networks listed in the `nets` attribute are saved according to the 
        network name (key in the `nets` attribute dictionary) and training 
        step: `<save_path>/<net_name>_<step>.pth`.

        We keep only the `keep` most recent iterations of the network weights.

        Parameters
        ----------
        save_path: str
            path to directory to save the network weights
        step: int
            current training step
        keep: int, default = 3
            number of most recent network weights to keep
        """
        if self.nets:
            for (name, net) in self.nets.items():
                dst = os.path.join(save_path, '%s_%i.pth' % (name, step))
                torch.save(net.state_dict(), dst)

            # Remove old files
            num_keep = keep * len(self.nets)
            weight_files = glob(os.path.join(save_path, "*.pth"))
            old_weight_files = sorted(weight_files, key=os.path.getctime)[
                :-num_keep
            ]
            for f in old_weight_files:
                os.remove(f)

    def load(self, load_path: str, save_num=-1) -> None:
        """
        Load the agent network(s) weights. Will load the most recent network 
        weight(s) from the `load_path` according to file creation time. Expects 
        weights in `*.pth` format.

        Parameters
        ----------
        load_path: str
            path to directory with network weights
        save_num: int
            number corresponding to a saved model. If -1, use most recent
        """

        path_str = (
            '%s*.pth' if save_num == -1 else '%s_' + str(save_num) + '.pth'
        )

        for (name, net) in self.nets.items():
            weight_files = glob(os.path.join(load_path, path_str % name))
            if len(weight_files) == 0:
                logging.error('Could not load weights from %s.' % load_path)
                exit()
            most_recent_weights = sorted(weight_files, key=os.path.getctime)[
                -1
            ]
            net.load_state_dict(torch.load(most_recent_weights))
            logging.info("Loaded %s." % most_recent_weights)

    def to_gpu(self) -> None:
        """
        Place agent `self.nets` on the GPU.
        """
        self.device = 'gpu'
        for (name, net) in self.nets.items():
            net = net.cuda()

    def to_cpu(self) -> None:
        """
        Place agent `self.nets` on the CPU.
        """
        self.device = 'cpu'
        for (name, net) in self.nets.items():
            net = net.cpu()

    def set_train(self) -> None:
        """
        Set agent `self.nets` to training mode.
        """
        self.mode = 'train'
        for (name, net) in self.nets.items():
            net = net.train()

    def set_test(self) -> None:
        """
        Set agent `self.nets` to evaluation mode.
        """
        self.mode = 'test'
        for (name, net) in self.nets.items():
            net = net.eval()
