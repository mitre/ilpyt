from typing import Dict, List

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity: int):
        """
        Experience replay memory to store transitions. Used in DQN.
        Adapted from:
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        Parameters
        ----------
        capacity: int
            maximum number of experiences to store in replay memory.
        """
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.next_states: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.rewards: List[float] = []

        self.size = 0
        self.capacity = capacity

    def add(self, batch: Dict[str, torch.Tensor]):
        """
        Add an experience to replay memory.

        Parameters
        ----------
        batch: dict
            single experience
        """
        self.states.append(batch['states'].to('cpu'))
        self.next_states.append(batch['next_states'].to('cpu'))
        self.actions.append(batch['actions'].to('cpu'))
        self.rewards.append(batch['rewards'].to('cpu'))
        self.dones.append(batch['dones'].to('cpu'))
        self.size += 1

        # Remove oldest experience if replay memory full
        if self.size > self.capacity:
            self.states.pop(0)
            self.next_states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.size -= 1

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample some transitions from replay memory.

        Parameters
        ----------
        batch_size: int
            number of experiences to sample from replay memory

        Returns
        -------
        Dict[str, torch.Tensor]:
            dictionary of random experiences if there are enough available, 
            else None
        """
        if batch_size > self.size:
            return None

        # Sample a batch
        idxs = np.random.randint(0, self.size, batch_size)
        states = [self.states[i] for i in idxs]
        next_states = [self.next_states[i] for i in idxs]
        actions = [self.actions[i] for i in idxs]
        rewards = [self.rewards[i] for i in idxs]
        dones = [self.dones[i] for i in idxs]

        sample_batch = {
            'states': torch.cat(states, dim=0),
            'actions': torch.cat(actions, dim=0),
            'rewards': torch.cat(rewards, dim=0),
            'next_states': torch.cat(next_states, dim=0),
            'dones': torch.cat(dones, dim=0),
        }

        return sample_batch

    def save(self, path: str) -> None:
        """
        Save the ReplayMemory buffer to a numpy file.

        Parameters
        ----------
        path: str
            save path for buffer array
        """
        b = np.asarray(self.buffer)
        np.save(path, b)

    def load(self, path: str) -> None:
        """
        Load a numpy file to the ReplayMemory buffer.

        Parameters
        ----------
        path: str
            load path for buffer array
        """
        b = np.load(path + '.npy', allow_pickle=True)
        assert b.shape[0] == self.memory_size

        for i in range(b.shape[0]):
            self.add(b[i])
