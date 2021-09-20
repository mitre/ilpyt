"""
Heuristic agents for various OpenAI Gym environments. The agent policies, in 
this case, are deterministic functions, and often handcrafted or found by 
non-gradient optimization algorithms, such as evolutionary strategies.

Many of the heuristic policies were adapted from the following source:
```
@book{xiao2022,
 title     = {Reinforcement Learning: Theory and {Python} Implementation},
 author    = {Zhiqing Xiao}
 publisher = {Springer Nature},
}
```
"""

from typing import Dict

import numpy as np
import torch

from ilpyt.agents.base_agent import BaseAgent


class LunarLanderContinuousHeuristicAgent(BaseAgent):
    """
    Heuristic policy for the OpenAI Gym LunarLanderContinuous-v2 environment.
    Adapted from the OpenAI Gym repository:
    https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
    """

    def initialize(self) -> None:
        """
        Pass. Heuristic agents do not require any initialization."
        """
        pass

    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Find best action for the given state.

        Parameters
        ----------
        state: torch.Tensor
            state tensor, of size (batch_size, 8) with attributes 
            [horizontal coordinate, vertical coordinate, horizontal speed, 
            vertical speed, angle, angular speed, first leg contact, 
            second leg contact]

        Returns
        -------
        np.ndarray:
            selected actions, of size (batch_size, 2)
        """
        batch_size = len(state)

        angle_targ = (
            state[:, 0] * 0.5 + state[:, 2] * 1.0
        )  # angle point towards center
        angle_targ = torch.clip(angle_targ, -0.4, 0.4)
        hover_targ = 0.55 * torch.abs(state[:, 0])  # target y proportional to
        # horizontal offset

        angle = (angle_targ - state[:, 4]) * 0.5 - (state[:, 5]) * 1.0
        hover = (hover_targ - state[:, 1]) * 0.5 - (state[:, 3]) * 0.5

        for i in range(batch_size):
            if state[i, 6] or state[i, 7]:  # legs have contact
                angle[i] = 0
                hover[i] = -(state[i, 3]) * 0.5  # override to reduce fall speed

        a = torch.stack([hover * 20 - 1, -angle * 20], dim=-1)
        a = torch.clamp(a, -1, +1)
        return a.cpu().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Pass. Heuristic agents do not update their agent policies.
        """
        return {}


class LunarLanderHeuristicAgent(BaseAgent):
    """
    Heuristic policy for the OpenAI Gym LunarLander-v2 environment.
    Adapted from the book 'Reinforcement Learning: Theory and Python Implementation':
    https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
    """

    def initialize(self):
        """
        Pass. Heuristic agents do not require any initialization."
        """
        pass

    def step(self, state: torch.Tensor):
        """
        Find best action for the given state.

        Parameters
        ----------
        state (torch.Tensor):
            state tensor, of size (batch_size, 8) with attributes 
            [horizontal coordinate, vertical coordinate, horizontal speed, 
            vertical speed, angle, angular speed, first leg contact, 
            second leg contact]

        Returns
        -------
        np.ndarray:
            selected actions, of size (batch_size, action_shape)
        """
        batch_size = len(state)

        angle_targ = (
            state[:, 0] * 0.5 + state[:, 2] * 1.0
        )  # angle point towards center
        angle_targ = torch.clip(angle_targ, -0.4, 0.4)
        hover_targ = 0.55 * torch.abs(state[:, 0])  # target y proportional to
        # horizontal offset

        angle = (angle_targ - state[:, 4]) * 0.5 - (state[:, 5]) * 1.0
        hover = (hover_targ - state[:, 1]) * 0.5 - (state[:, 3]) * 0.5

        for i in range(batch_size):
            if state[i, 6] or state[i, 7]:  # legs have contact
                angle[i] = 0
                hover[i] = -(state[i, 3]) * 0.5  # override to reduce fall speed

        a = np.zeros(batch_size, dtype=np.uint8)
        for i in range(batch_size):
            if hover[i] > torch.abs(angle[i]) and hover[i] > 0.05:
                a[i] = 2
            elif angle[i] < -0.05:
                a[i] = 3
            elif angle[i] > +0.05:
                a[i] = 1
        return a

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Pass. Heuristic agents do not update their agent policies.
        """
        return {}


class CartPoleHeuristicAgent(BaseAgent):
    """
    Heuristic agent for the OpenAI Gym CartPole-v0 environment.
    Adapted from the book 'Reinforcement Learning: Theory and Python Implementation':
    https://github.com/ZhiqingXiao/OpenAIGymSolution
    """

    def initialize(self):
        """
        Pass. Heuristic agents do not require any initialization."
        """
        pass

    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Find best action for the given state. The overall policy followed by the 
        CartPole agent: push right when 3*angle + angle_velocity > 0.

        Parameters
        ----------
        state: torch.Tensor
            state tensor of size (batch_size, 4) with attributes 
            [cart position, cart velocity, pole angle, pole velocity at tip]

        Returns
        -------
        np.ndarray:
            action, of shape (batch_size, ) where 0= push cart to left, 1 = push cart to right
        """
        angle, angle_velocity = state[:, 2], state[:, 3]
        a = (3 * angle + angle_velocity) > 0
        return a.cpu().long().numpy()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Pass. Heuristic agents do not update their agent policies.
        """
        return {}


class MountainCarHeuristicAgent(BaseAgent):
    """
    Fixed deterministic policy for the OpenAI gym MountainCar-v0 environment.
    Adapted from the book 'Reinforcement Learning: Theory and Python Implementation':
    https://github.com/ZhiqingXiao/OpenAIGymSolution
    """

    def initialize(self):
        """
        Pass. Heuristic agents do not require any initialization."
        """
        pass

    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Find best action for the given state. Push right when satisfying a 
        certain condition; otherwise push left.

        Parameters
        ----------
        state: torch.Tensor
            state tensor of size (batch_size, 2) with attributes 
            [position, velocity]
        Returns
        -------
        np.ndarray:
            discrete action of shape (batch_size, ) where 
            0 = push left, 1 = no push, 2 = push right
        """
        actions = []
        positions, velocities = state[:, 0], state[:, 1]
        for (position, velocity) in zip(positions, velocities):
            lb = min(
                -0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008,
            )
            ub = -0.07 * (position + 0.38) ** 2 + 0.07
            if lb < velocity < ub:
                action = 2  # push right
            else:
                action = 0  # push left
            actions.append(action)
        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Pass. Heuristic agents do not update their agent policies.
        """
        return {}


class MountainCarContinuousHeuristicAgent(BaseAgent):
    """
    Heuristic agent for the OpenAI Gym MountainCarContinuous-v0 environment.
    Adapted from the book 'Reinforcement Learning: Theory and Python Implementation':
    https://github.com/ZhiqingXiao/OpenAIGymSolution
    """

    def initialize(self):
        """
        Pass. Heuristic agents do not require any initialization."
        """
        pass

    def step(self, state: torch.Tensor) -> np.ndarray:
        """
        Find best action for the given state. Push right when satisfying a 
        certain condition; otherwise push left.

        Parameters
        ----------
        state: torch.Tensor
            state tensor of size (batch_size, 2) with attributes 
            [position, velocity]

        Returns
        -------
        np.ndarray:
            continuous action of shape (batch_size, ) - pushing the car to the 
            left or to the right
        """
        positions, velocities = state[:, 0], state[:, 1]
        actions = []
        for (position, velocity) in zip(positions, velocities):
            if position > -4 * velocity or position < 13 * velocity - 0.6:
                force = 1.0
            else:
                force = -1.0
            actions.append(
                [
                    force,
                ]
            )
        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Pass. Heuristic agents do not update their agent policies.
        """
        return {}
