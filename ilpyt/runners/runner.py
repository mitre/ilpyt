"""
The runner coordinates the agent-environment interaction loop.  It collects 
transitions (state, action, reward, next state) over specified intervals of 
time.  We can have the runner generate a collection of transitions for us by 
calling `generate_batch` and `generate_episodes`. 
"""

from typing import Any, Dict, List

import numpy as np
import torch

from ilpyt.agents.base_agent import BaseAgent
from ilpyt.envs.vec_env import VecEnv
from ilpyt.utils.seed_utils import set_seed


class Experiences:
    def __init__(self) -> None:
        """
        Initialize experiences object, which stores a stack of agent-environment 
        transitions.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Add a transition to the stack of transitions.

        Parameters
        ----------
        state: torch.Tensor
        action: torch.Tensor
        reward: torch.Tensor
        done: torch.Tensor
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def to_torch(self) -> None:
        """
        Convert the stack of transitions from a list of torch.Tensors to a 
        single instance of torch.Tensor.
        """
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.rewards = torch.tensor(self.rewards)
        self.dones = torch.tensor(self.dones)

    def to_gpu(self) -> None:
        """
        Place the experience on the GPU.
        """
        self.states = self.states.cuda()
        self.actions = self.actions.cuda()
        self.rewards = self.rewards.cuda()
        self.dones = self.dones.cuda()

    def get_episode_rewards(self) -> List[float]:
        """
        Get the episode rewards.

        Returns
        -------
        List[float]:
            list of episode rewards
        """
        cumulative_rewards = []
        episode_ends = torch.where(self.dones)[0] + 1
        for i in range(len(episode_ends)):
            if i == 0:
                start = 0
            else:
                start = episode_ends[i - 1]
            end = episode_ends[i]
            r = torch.sum(self.rewards[start:end]).item()
            cumulative_rewards.append(r)
        return cumulative_rewards


class Runner:
    def __init__(self, env: VecEnv, agent: BaseAgent, use_gpu: bool) -> None:
        """
        The runner manages the agent and environment interaction to collect
        episodes and/or rollouts.

        Parameters
        ----------
        env: VecEnv
            Multiprocessing compatible gym environment
        agent: BaseAgent
            Agent to collect rollouts or episode experiences from
        use_gpu: bool
            whether or not to use GPU, if false use CPU
        """
        self.env = env
        self.agent = agent
        self.num_env = self.env.num_envs
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.agent.to_gpu()

        # Initialize state
        self.state = torch.tensor(self.env.reset())
        if self.use_gpu:
            self.state = self.state.cuda()

        # Episode statistics
        # Each list entry corresponds to a different parallel environment.
        self.episode_stats = {
            'reward': np.zeros(self.num_env),
            'length': np.zeros(self.num_env),
            'count': np.zeros(self.num_env),
        }

    def reset(self) -> None:
        """
        Reset the state and episode stats within the Runner.
        """
        self.state = torch.tensor(self.env.reset())
        if self.use_gpu:
            self.agent.to_gpu()
            self.state = self.state.cuda()
        self.episode_stats = {
            'reward': np.zeros(self.num_env),
            'length': np.zeros(self.num_env),
            'count': np.zeros(self.num_env),
        }

    @torch.no_grad()
    def generate_batch(self, rollout_steps: int) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of rollouts.

        Will return a dictionary with keys: states, next_states, actions,
        rewards, dones, and infos.

        - states and next_states will have a shape of (rollout_steps, num_env, state_shape).
        - actions will have a shape of (rollout_steps, num_env, act_shape).
        - rewards and dones will have a shape of (rollout_steps, num_env).
        - infos will contain episode metadata -- it will be expressed as a list of dictionaries with values summarizing the episode_length and total_reward accumulated.

        Parameters
        ----------
        rollout_steps: int
            number of rollout steps to collect

        Returns
        -------
        Dict[str, torch.Tensor]:
            batch of rollouts with keys: states, next_states, actions, rewards, 
            dones, and infos
        """
        # Initialize batch
        batch_size = (
            rollout_steps,
            self.num_env,
        )
        obs_shape = batch_size + self.env.observation_shape
        if self.env.type == 'discrete':
            act_shape = batch_size
        else:
            act_shape = batch_size + self.env.action_shape
        batch: Dict[str, Any] = {
            'states': torch.empty(obs_shape),
            'next_states': torch.empty(obs_shape),
            'actions': torch.empty(act_shape),
            'rewards': torch.empty(batch_size),
            'dones': torch.empty(batch_size),
            'infos': [],
        }

        for step in range(rollout_steps):
            # Agent takes action
            action = self.agent.step(self.state)

            # Update environment
            next_state, reward, done, info = self.env.step(action)
            # print('REWARD IN RUNNER:' , reward)
            # Record transition to batch
            batch['states'][step] = torch.as_tensor(self.state)
            batch['next_states'][step] = torch.as_tensor(next_state)
            batch['actions'][step] = torch.tensor(
                action, dtype=torch.float, requires_grad=True
            )
            batch['rewards'][step] = torch.as_tensor(reward)
            batch['dones'][step] = torch.as_tensor(done)

            # Update episode stats
            self.episode_stats['reward'] += reward
            self.episode_stats['length'] += np.ones(self.num_env)
            self.episode_stats['count'] += done

            # On episode end, update batch infos and reset
            for i in range(self.num_env):
                if done[i]:
                    update_dict = {
                        'reward/%i' % i: self.episode_stats['reward'][i],
                        'length/%i' % i: self.episode_stats['length'][i],
                    }
                    update = [self.episode_stats['count'][i], update_dict]
                    batch['infos'].append(update)
                    self.episode_stats['reward'][i] = 0
                    self.episode_stats['length'][i] = 0

            # Update state
            self.state = torch.tensor(next_state)
            if self.use_gpu:
                self.state = self.state.cuda()

        # Batch to GPU
        if self.use_gpu:
            for (k, v) in batch.items():
                if k != 'infos':
                    batch[k] = v.cuda()

        return batch

    @torch.no_grad()
    def generate_episodes(self, num_episodes: int) -> Experiences:
        """
        Generate episodes.
        Only records states, actions, rewards.

        Will return a list of torch Tensors.

        Parameters
        ----------
        num_episodes: int
            number of episodes to collectively acquire across all of the
            environment threads

        Returns
        -------
        Experiences (Dict[str, torch.Tensor]]):
            {'states': [], 'actions': [], 'rewards': [], 'dones': []}
        """
        # Initialize batch
        eps_by_env = [Experiences() for i in range(self.num_env)]
        all_episodes = []

        ep_count = 0
        self.env.reset()
        while ep_count < num_episodes:

            # Agent takes action
            action = self.agent.step(self.state)

            # Update environment
            next_state, reward, done, info = self.env.step(action)

            # Record transition to batch
            # On episode end, update batch infos and reset
            for i in range(self.num_env):
                # Record transition to buffer
                eps_by_env[i].add(
                    torch.as_tensor(self.state[i]),
                    torch.as_tensor(action[i]),
                    torch.as_tensor(reward[i]),
                    torch.as_tensor(done[i]),
                )

                # On episode end, move from buffer to result_dict
                if done[i]:
                    all_episodes.append(eps_by_env[i])
                    next_state[i] = self.env.envs[i].reset()
                    eps_by_env[i] = Experiences()
                    ep_count += 1
                    if ep_count >= num_episodes:
                        break

            # Update state
            self.state = torch.tensor(next_state)
            if self.use_gpu:
                self.state = self.state.cuda()

        # Combine experiences across all environments
        eps = Experiences()
        for i in range(len(all_episodes)):
            eps.states += all_episodes[i].states
            eps.actions += all_episodes[i].actions
            eps.rewards += all_episodes[i].rewards
            eps.dones += all_episodes[i].dones
        eps.to_torch()
        if self.use_gpu:
            eps.to_gpu()

        return eps

    @torch.no_grad()
    def generate_test_episodes(
        self, num_episodes: int, start_seed=24
    ) -> Experiences:
        """
        Generate episodes using a single env with seeds for reproducibility.
        Only records states, actions, rewards.

        Will return a list of torch Tensors.

        Use for testing when you need to compare against other algorithms or runs.

        Parameters
        ----------
        num_episodes: int
            number of episodes to collectively acquire across all of the
            environment threads

        Returns
        -------
        Experiences (Dict[str, torch.Tensor]]):
            {'states': [], 'actions': [], 'rewards': [], 'dones': []}
        """
        # Initialize batch
        eps = Experiences()

        test_env = self.env.envs[0]

        ep_count = 0
        test_env.seed(start_seed * (ep_count + 1))
        set_seed(start_seed * (ep_count + 1))
        test_state = torch.tensor(
            test_env.reset().copy(), dtype=torch.float
        ).unsqueeze(0)

        if self.use_gpu:
            test_state = test_state.cuda()

        while ep_count < num_episodes:

            # Agent takes action
            action = self.agent.step(test_state)

            # Update environment
            next_state, reward, done, info = test_env.step(action[0])

            # Record transition to batch
            # On episode end, update batch infos and reset
            eps.add(
                torch.as_tensor(test_state.squeeze()),
                torch.as_tensor(action.squeeze()),
                torch.as_tensor(reward),
                torch.as_tensor(done),
            )

            if done:
                ep_count += 1
                test_env.seed(start_seed * (ep_count + 1))
                set_seed(start_seed * (ep_count + 1))
                test_state = torch.tensor(
                    test_env.reset().copy(), dtype=torch.float
                ).unsqueeze(0)
            else:
                # Update state
                test_state = torch.tensor(
                    next_state.copy(), dtype=torch.float
                ).unsqueeze(0)

            if self.use_gpu:
                test_state = test_state.cuda()

        eps.to_torch()
        if self.use_gpu:
            eps.to_gpu()

        return eps
