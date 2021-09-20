"""
Example of how to use the ray tune library to perform hyperparameter sweeps on 
the Proximal Policy Optimization (PPO) algorithm.
"""
import argparse
from functools import partial

import numpy as np
import pandas as pd
from ray import tune
from ray.tune import Analysis, CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.rl import RL
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.utils.seed_utils import set_seed

pd.set_option(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.max_colwidth",
    None,
)


def train(
    config,
    env_id,
    num_episodes,
    num_env: int = 16,
    use_gpu: bool = True,
    seed: int = 24,
):
    # Set random seed
    set_seed(seed)

    # Build environment
    env = build_env(env_id=env_id, num_env=num_env, seed=seed)

    # Build agent
    agent = PPOAgent(
        actor=choose_net(env, activation='tanh'),
        critic=choose_net(env, activation='tanh', output_shape=1),
        lr=config['lr'],
        gamma=config['gamma'],
        clip_ratio=config['clip_ratio'],
        entropy_coeff=config['entropy_coeff'],
    )

    algo = RL(
        env=env, agent=agent, use_gpu=use_gpu, save_path='.', load_path=''
    )
    algo.train(
        num_episodes=num_episodes, rollout_steps=config['rollout_steps']
    )
    tune.report(reward=np.mean(algo.reward_tracker))


def hyperparameter_search(env, num_samples=100, max_num_epochs=5000):
    config = {
        "clip_ratio": tune.choice([0.1, 0.2, 0.3]),
        "gamma": tune.choice([0.9, 0.99, 0.999]),
        "lr": tune.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "rollout_steps": tune.choice([8, 16, 32, 64]),
        "entropy_coeff": tune.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    }

    scheduler = ASHAScheduler(
        metric="reward",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(metric_columns=["reward", "training_iteration"])

    result = tune.run(
        partial(train, env_id=env, num_episodes=max_num_epochs),
        name="PPO_%s" % env,
        resources_per_trial={"cpu": 1, "gpu": 0.1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        raise_on_failed_trial=False,
    )

    best_trial = result.get_best_trial("reward", "max", "last-5-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial reward: {}".format(best_trial.last_result["reward"]))


def report_results(results_dir):
    result = Analysis(results_dir)
    trials = result.dataframe("reward", "max")
    top_trials = trials.sort_values(by=['reward']).tail(5)
    selected_columns = [col for col in top_trials.columns if 'config' in col]
    selected_columns += ['reward', 'done', 'logdir']
    print(top_trials[selected_columns])
    top_trials[selected_columns].to_csv('results.csv')


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='LunarLanderContinuous-v2',
        help='name of registered gym environment',
    )

    args = parser.parse_args()

    hyperparameter_search(args.env)
