"""
Example of how to train the Behavioral Cloning (BC) algorithm from scratch. 

Also includes notes on how to resume training from an earlier checkpoint, 
perform testing/evaluation, and run the baselines from the model_zoo. 
"""

import logging
import os

from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.bc import BC
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
    agent = ImitationAgent(net=net, lr=0.0001)

    # Build algorithm
    algo = BC(
        agent=agent,
        env=env,
        use_gpu=use_gpu,
        save_path=save_path,
        load_path=load_path,
    )
    return algo


def evaluate_baselines():
    envs = [
        'MountainCar-v0',
        'CartPole-v0',
        'LunarLander-v2',
        'MountainCarContinuous-v0',
        'LunarLanderContinuous-v2',
    ]

    for env in envs:
        logging.debug(env)
        save_path = os.path.join('logs/BC/', env)
        load_path = os.path.join('/mnt/IMLEARN/model_zoo/BC/', env)

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
    #       save_path = 'dir/to/save/to/
    #       load_path = ''
    # To continue training a model:
    #       save_path = 'dir/to/save/to/
    #       load_path = 'dir/to/load/from/'
    # To test an old model:
    #       Comment out train() method and algo.agent.load()
    #       save_path = 'dir/to/save/to/'
    #       load_path = 'dir/to/load/from/'

    if not os.path.exists('logs/BC'):
        os.makedirs('logs/BC')
    evaluate_baselines()
    # LunarLander-v2 -----------------------------------------------------------

    save_path = 'logs/BC/LunarLander-v2'
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
        num_epochs=10000,
        batch_size=20,
        expert_demos='demos/LunarLander-v2/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # CartPole-v0 --------------------------------------------------------------

    save_path = 'logs/BC/CartPole-v0'
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
        num_epochs=10000,
        batch_size=20,
        expert_demos='demos/CartPole-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MountainCar-v0 -----------------------------------------------------------

    save_path = 'logs/BC/MountainCar-v0'
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
        num_epochs=10000,
        batch_size=20,
        expert_demos='demos/MountainCar-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MountainCarContinuous-v0 ----------------------------------------------------------------

    save_path = 'logs/BC/MountainCarContinuous-v0'
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
        num_epochs=10000,
        batch_size=20,
        expert_demos='demos/MountainCarContinuous-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # LunarLanderContinuous-v0 -------------------------------------------------

    save_path = 'logs/BC/LunarLanderContinuous-v2'
    load_path = ''

    # Build experiment -----
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='LunarLanderContinuous-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_epochs=10000,
        batch_size=20,
        expert_demos='demos/LunarLanderContinuous-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)
