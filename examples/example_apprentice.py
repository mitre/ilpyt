"""
Example of how to train the Apprenticeship learning (AppL) algorithm from 
scratch. 

Also includes notes on how to resume training from an earlier checkpoint, 
perform testing/evaluation, and run the baselines from the model_zoo. 
"""

import logging
import os

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.algos.apprentice import Apprentice
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.utils.seed_utils import set_seed


def build(
    agent_type: str,
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
    if agent_type == 'a2c':
        agent = A2CAgent(
            actor=choose_net(env),
            critic=choose_net(env, output_shape=1),
            lr=7e-5,
            entropy_coeff=1e-5,
        )
    elif agent_type == 'dqn':
        agent = DQNAgent(
            net=choose_net(env),
            target_net=choose_net(env),
            num_actions=env.num_actions,
            num_envs=num_env,
        )

    # Build learner
    algo = Apprentice(
        env=env,
        agent=agent,
        use_gpu=use_gpu,
        save_path=save_path,
        load_path=load_path,
    )
    return algo


def evaluate_baselines():
    envs = ['MountainCar-v0', 'CartPole-v0']

    for env in envs:
        logging.debug(env)
        save_path = os.path.join('logs/Apprentice/', env)
        load_path = os.path.join('model_zoo/Apprentice/', env)

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

    if not os.path.exists('logs/Apprentice'):
        os.makedirs('logs/Apprentice')

    # CartPole-v0 --------------------------------------------------------------

    save_path = 'logs/Apprentice/CartPole-v0/'
    load_path = ''

    # Build experiment
    algo = build(
        agent_type='a2c',
        save_path=save_path,
        load_path=load_path,
        env_id='CartPole-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=100,
        epsilon=0.00001,
        gamma=0.9,
        batch_size=64,
        num_train=10000,
        expert_demos='demos/CartPole-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MountainCar-v0 -----------------------------------------------------------

    save_path = 'logs/Apprentice/MountainCar-v0/'
    load_path = ''

    # Build experiment
    algo = build(
        agent_type='dqn',
        save_path=save_path,
        load_path=load_path,
        env_id='MountainCar-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=100,
        epsilon=0.00001,
        gamma=0.9,
        batch_size=64,
        num_train=10000,
        expert_demos='demos/MountainCar-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)
