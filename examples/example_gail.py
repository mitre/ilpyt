"""
Example of how to train the Generative Adversarial Imitation Learning (GAIL) 
algorithm from scratch. 

Also includes notes on how to resume training from an earlier checkpoint, 
perform testing/evaluation, and run the baselines from the model_zoo. 
"""

import logging
import os

from ilpyt.agents.gail_agent import GAILAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.gail import GAIL
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
    generator = PPOAgent(
        actor=choose_net(env, activation='tanh'),
        critic=choose_net(env, activation='tanh', output_shape=1),
        lr=0.0005,
        gamma=0.99,
        clip_ratio=0.1,
        entropy_coeff=0.01,
    )
    agent = GAILAgent(
        gen=generator,
        disc=choose_net(
            env, output_shape=1, activation='tanh', with_action_shape=True
        ),
        lr=0.00065,
    )

    # Build learner
    algo = GAIL(
        env=env,
        agent=agent,
        use_gpu=use_gpu,
        save_path=save_path,
        load_path=load_path,
    )

    return algo


def evaluate_baselines():
    envs = [
        'MountainCar-v0',
        'MountainCarContinuous-v0',
        'CartPole-v0',
        'LunarLander-v2',
        'LunarLanderContinuous-v0',
    ]

    for env in envs:
        logging.debug(env)
        save_path = os.path.join('logs/GAIL/', env)
        load_path = os.path.join('model_zoo/GAIL/', env)

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

    if not os.path.exists('logs/GAIL'):
        os.makedirs('logs/GAIL')

    # LunarLander-v2 -----------------------------------------------------------

    save_path = 'logs/GAIL/LunarLander-v2/'
    load_path = ''

    # Build experiment
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='LunarLander-v2',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=10000,
        rollout_steps=64,
        expert_demos='demos/LunarLander-v2/demos.pkl',
        save_interval=1000,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # LunarLander-v2 -----------------------------------------------------------

    save_path = 'logs/GAIL/LunarLanderContinuous-v2/'
    load_path = ''

    # Build experiment
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='LunarLanderContinuous-v2',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=10000,
        rollout_steps=64,
        expert_demos='demos/LunarLanderContinuous-v2/demos.pkl',
        save_interval=1000,
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # CartPole-v0 --------------------------------------------------------------

    save_path = 'logs/GAIL/CartPole-v0/'
    load_path = ''

    # Build experiment
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='CartPole-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=10000,
        rollout_steps=64,
        expert_demos='demos/CartPole-v0/demos.pkl',
    )

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MoutainCar-v0 ------------------------------------------------------------

    save_path = 'logs/GAIL/MountainCar-v0/'
    load_path = ''

    # Build experiment
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='MountainCar-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=10000,
        rollout_steps=64,
        expert_demos='demos/MoutainCar-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)

    # MoutainCarContinuous-v0 --------------------------------------------------

    save_path = 'logs/GAIL/MountainCarContinuous-v0/'
    load_path = ''

    # Build experiment
    algo = build(
        save_path=save_path,
        load_path=load_path,
        env_id='MountainCarContinuous-v0',
        num_env=16,
        use_gpu=True,
    )

    # Train
    algo.train(
        num_episodes=10000,
        rollout_steps=64,
        expert_demos='demos/MountainCarContinuous-v0/demos.pkl',
    )
    # Close training environmnet
    algo.env.close()

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)
