"""
An implementation of a behavioral cloning (BC) algorithm, as in An Autonomous Land 
Vehicle in a Neural Network (ALVINN). The BC algorithm was described in the 
paper "An Autonomous Land Vehicle in a Neural Network" by Dean A. Pomerleau, and 
presented at NIPS 1988.

For more details, please refer to the paper: https://papers.nips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf
"""

import logging
import pickle

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.base_algo import BaseAlgorithm
from ilpyt.envs.vec_env import VecEnv
from ilpyt.runners.runner import Runner


class BC(BaseAlgorithm):
    def initialize(
        self,
        env: VecEnv,
        agent: BaseAgent,
        save_path: str = 'logs',
        load_path: str = '',
        use_gpu: bool = True,
    ) -> None:
        """
        Initialization function for the BC algorithm.

        Parameters
        ----------
        env: VecEnv
            vectorized OpenAI Gym environment
        agent: BaseAgent
            agent for train and/or test
        save_path: str, default='logs'
            path to directory to save network weights
        load_path: str, default=''
            path to directory to load network weights. If not specified, network 
            weights will be randomly initialized
        use_gpu: bool, default=True
            flag indicating whether or not to run operations on the GPU

        Raises
        ------
        ValueError:
            if `agent` is not an instance of `ImitationAgent`
        """
        self.env = env
        self.agent = agent
        self.use_gpu = use_gpu

        if not isinstance(agent, ImitationAgent):
            raise ValueError(
                'Behavioral cloning is only compatible with ImitationAgents.'
            )
        if use_gpu:
            self.agent.to_gpu()
        if load_path:
            self.agent.load(load_path)

        # Set up runner
        self.runner = Runner(env, self.agent, use_gpu)

    def train(
        self,
        num_epochs: int = int(1e4),
        batch_size: int = 20,
        expert_demos: str = 'demos.pkl',
    ) -> None:
        """
        Train the agent in the specified environment.

        Parameters
        ----------
        num_epochs: int, default=1e4
            number training epochs
        batch_size: int, default=20
            batch size
        expert_demos: str, default='demos.pkl'
            path to expert demonstrations file, expects a *.pkl of a 
            `runner.Experiences` object
        """
        # Expert demonstrations
        with open(expert_demos, 'rb') as f:
            demos = pickle.load(f)  # runner.Experiences
            if self.use_gpu:
                demos.to_gpu()
        dataset = TensorDataset(demos.states, demos.actions)

        self.best_loss = np.float('inf')

        self.agent.set_train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train learner on expert dataset
        step = 0
        for epoch in range(num_epochs):
            for i, (states, actions) in enumerate(loader):

                # Step agent
                self.agent.step(states)

                # Update agent
                batch = {'states': states, 'actions': actions}
                loss_dict = self.agent.update(batch)

                # Log
                self.log(step, loss_dict)
                step += 1

            # Save agent
            loss = loss_dict['loss/total']
            logging.debug('Epoch %i, Loss %0.4f' % (epoch, loss))
            if loss < self.best_loss:
                self.agent.save(self.save_path, step)
                self.best_loss = loss
                logging.info(
                    "Save new best model at epoch %i with loss %0.4f."
                    % (epoch, loss)
                )

