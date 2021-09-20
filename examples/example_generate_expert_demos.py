"""
Example of how to generate expert demonstrations using the provided heuristic 
agents. 
"""

import argparse
import pickle

import numpy as np
import torch

import ilpyt.agents.heuristic_agent as heuristic
from ilpyt.utils.env_utils import build_env
from ilpyt.runners.runner import Experiences, Runner


def generate_demos(expert, env_id, num_env, num_episodes, threshold):
    # Build environment
    env = build_env(env_id=env_id, num_env=num_env, seed=24)

    # Gather demonstrations
    demos = Experiences()
    i = 0
    while i <= num_episodes:
        runner = Runner(env, expert, use_gpu=False)
        demo = runner.generate_episodes(1)
        if torch.sum(demo.rewards) < threshold:
            continue
        else:
            demos.states += demo.states
            demos.actions += demo.actions
            demos.rewards += demo.rewards
            demos.dones += demo.dones
            i += 1

    demos.to_torch()
    rewards = demos.get_episode_rewards()
    print(
        demos.states.shape,
        demos.actions.shape,
        demos.rewards.shape,
        demos.dones.shape,
    )
    print("Average reward:", np.mean(rewards))
    print("Standard deviation:", np.std(rewards))

    return demos


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BipedalWalker-v3')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument(
        '--save_path',
        type=str,
        default='demos/demos.pkl',
        help='file path to save trained models and results',
    )
    args = parser.parse_args()

    # Generate demonstrations
    if args.env_name == 'LunarLander-v2':
        expert = heuristic.LunarLanderHeuristicAgent()
        threshold = 200
    elif args.env_name == 'LunarLanderContinuous-v2':
        expert = heuristic.LunarLanderContinuousHeuristicAgent()
        threshold = 200
    elif args.env_name == 'CartPole-v0':
        expert = heuristic.CartPoleHeuristicAgent()
        threshold = 195
    elif args.env_name == 'MountainCar-v0':
        expert = heuristic.MountainCarHeuristicAgent()
        threshold = -110
    elif args.env_name == 'MountainCarContinuous-v0':
        expert = heuristic.MountainCarContinuousHeuristicAgent()
        threshold = 90

    demos = generate_demos(
        expert, args.env_name, 1, args.num_episodes, threshold
    )

    # Save demonstrations
    with open(args.save_path, 'wb') as f:
        pickle.dump(demos, f, pickle.HIGHEST_PROTOCOL)
