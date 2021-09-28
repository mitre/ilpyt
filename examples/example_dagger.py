"""
Example of how to train the Dataset Aggregation (DAgger) algorithm from scratch. 

Also includes notes on how to resume training from an earlier checkpoint, 
perform testing/evaluation, and run the baselines from the model_zoo. 
"""

import logging
import os

import ilpyt.agents.heuristic_agent as heuristic
from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.dagger import DAgger
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.utils.seed_utils import set_seed


def build(
    save_path: str,
    load_path: str,
    env_id: str,
    num_env: int,
    use_gpu: bool,
    seed: int = 24,
):
    # Set random seed
    set_seed(seed)

    # Build environment
    env = build_env(env_id=env_id, num_env=num_env, seed=seed)

    # Build agent
    net = choose_net(env)
    agent = ImitationAgent(net=net, lr=0.00005)

    # Build learner
    algo = DAgger(
        env=env,
        agent=agent,
        use_gpu=use_gpu,
        save_path=save_path,
        load_path=load_path,
        max_mem=int(1e4),
    )
    return algo


def evaluate_baselines():
    envs = [
        'LunarLander-v2',
        'LunarLanderContinuous-v2',
        'MountainCar-v0',
        'MountainCarContinuous-v0',
        'CartPole-v0',
    ]

    for env in envs:
        logging.debug(env)
        save_path = os.path.join('logs/DAgger/', env)
        load_path = os.path.join('model_zoo/DAgger/', env)

        # Build experiment -----
        algo = build(
            save_path=save_path,
            load_path=load_path,
            env_id=env,
            num_env=16,
            use_gpu=True,
        )

        algo.env.close()

        algo.test(num_episodes=100)


if __name__ == '__main__':

    # NOTE
    # To train a new model:
    #       save_path = 'dir/to/save/to/'
    #       load_path = ''
    # To continue training a model:
    #       save_path = 'dir/to/save/to/'
    #       load_path = 'dir/to/load/from/'
    # To test an old model:
    #       Comment out train() method and algo.agent.load()
    #       save_path = 'dir/to/save/to/'
    #       load_path = 'dir/to/load/from/'

    if not os.path.exists('logs/DAgger'):
        os.makedirs('logs/DAgger')

    # LunarLander-v2 -----------------------------------------------------------

    save_path = 'logs/DAgger/LunarLander-v2/'
    load_path = ''

    # Build experiment -----
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='LunarLander-v2',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        expert=heuristic.LunarLanderHeuristicAgent(),
        num_episodes=100,
        num_epochs=100,
        batch_size=32,
        T_steps=20,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # CartPole-v0 --------------------------------------------------------------

    save_path = 'logs/DAgger/CartPole-v0/'
    load_path = ''

    # Build experiment -----
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='CartPole-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        expert=heuristic.CartPoleHeuristicAgent(),
        num_episodes=100,
        num_epochs=100,
        batch_size=20,
        T_steps=20,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MountainCar-v0 -----------------------------------------------------------

    save_path = 'logs/DAgger/MountainCar-v0/'
    load_path = ''

    # Build experiment -----
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='MountainCar-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        expert=heuristic.MountainCarHeuristicAgent(),
        num_episodes=100,
        num_epochs=100,
        batch_size=20,
        T_steps=20,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # LunarLanderContinuous-v2 -------------------------------------------------

    save_path = 'logs/DAgger/LunarLanderContinuous-v2/'
    load_path = ''

    # Build experiment -----
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='LunarLanderContinuous-v2',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        expert=heuristic.LunarLanderContinuousHeuristicAgent(),
        num_episodes=100,
        num_epochs=100,
        batch_size=20,
        T_steps=20,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MountainCarContinuous-v0 -------------------------------------------------

    save_path = 'logs/DAgger/MountainCarContinuous-v0/'
    load_path = ''

    # Build experiment -----
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='MountainCarContinuous-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        expert=heuristic.MountainCarContinuousHeuristicAgent(),
        num_episodes=100,
        num_epochs=100,
        batch_size=20,
        T_steps=20,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)
