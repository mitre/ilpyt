"""
`BaseAlgorithm` is the abstract class for an algorithm. An algorithm's role 
during learning is to coordinate the agent and environment during `train` and 
`test` time. 

To create a custom algorithm, simply extend `BaseAlgorithm`. The `BaseAlgorithm` 
API requires you to override the `initialize` and `train` methods.

- `initalize` sets algorithm class variables, such as the agent, environment, 
and possibly expert instances.
- `train` initiates agent learning in an environment.
- `test` evaluates agent performance in an environment.
"""
import logging
import os
import sys
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

import coloredlogs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.envs.vec_env import VecEnv


class BaseAlgorithm(metaclass=ABCMeta):
    def __init__(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        **kwargs:
            arbitrary keyword arguments. Will be passed to the `initialize` and 
            `setup_experiment` functions
        """
        self.setup_experiment(**kwargs)
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, env: VecEnv, agent: BaseAgent) -> None:
        """
        Perform algorithm initialization. This could include setting class 
        variables for agents, environments, and experts. Override this method 
        when extending the `BaseAlgorithm` class.

        Parameters
        ----------
        env: VecEnv
            vectorized OpenAI Gym environment
        agent: BaseAgent
            agent for train and/or test
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the algorithm. Override this method when extending the 
        `BaseAlgorithm` class.
        """
        pass

    @torch.no_grad()
    def test(self, num_episodes: int = 100) -> List:
        """
        Test the algorithm.

        After testing completes, test results (as a *.npy file) will be saved 
        to `self.save_path`.

        Parameters
        ----------
        num_episodes: int, default=100
            number of test episodes

        Returns
        -------
        List:
            list of test episode rewards
        """
        # Set agents to test
        assert hasattr(self, 'runner')
        assert hasattr(self, 'agent')

        self.agent.set_test()

        # Run episodes
        test_episodes = self.runner.generate_test_episodes(
            num_episodes, start_seed=24
        )
        rewards = test_episodes.get_episode_rewards()

        # Save results
        results_path = os.path.join(self.save_path, 'test_results.npy')
        np.save(results_path, rewards)

        logging.info('Testing complete!')
        logging.info('Results saved to %s.' % results_path)
        logging.info('Average reward: %0.3f' % np.mean(rewards))
        logging.info('Average standard deviation: %0.3f' % np.std(rewards))

        # Close environment
        self.env.close()

        return rewards

    def setup_experiment(self, save_path: str, **kwargs: Any) -> None:
        """
        Set up the algorithm save directory and logging (to TensorBoard and 
        to terminal).

        Parameters
        ----------
        save_path: str
            path to directory to save results, logs, network weights
        **kwargs:
            arbitrary keyword arguments
        """
        # Setup results directory
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path

        # Set up writer
        self.writer = SummaryWriter(self.save_path)

        # Set up logging
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        coloredlogs.install(level='DEBUG')

    def log(self, step: int, record_dict: Dict[str, float]) -> None:
        """
        Record information from `record_dict` in TensorBoard.

        Parameters
        ----------
        step: int
            training step
        record_dict: Dict[str, float]
            dictionary of values to record in TensorBoard, where keys correspond 
            to string names and values correspond to values to record
        """
        for name, value in record_dict.items():
            self.writer.add_scalar(name, value, step)
