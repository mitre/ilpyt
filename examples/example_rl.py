"""
Example of how to train the Proximal Policy Optimization (PPO), Advantage Actor
Critic (A2C), and Deep Q-Network (DQN) algorithms from scratch. 

Also includes notes on how to resume training from an earlier checkpoint, 
perform testing/evaluation, and run the baselines from the model_zoo. 
"""
import logging
import os

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.rl import RL
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.utils.seed_utils import set_seed


def build(
    rl_algorithm: str,
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
    if rl_algorithm == 'DQN':
        agent = DQNAgent(
            net=choose_net(env),
            target_net=choose_net(env),
            num_actions=env.num_actions,
            num_envs=num_env,
        )
    elif rl_algorithm == 'A2C':
        agent = A2CAgent(
            actor=choose_net(env),
            critic=choose_net(env, output_shape=1),
            lr=7e-5,
            entropy_coeff=1e-5,
        )
    elif rl_algorithm == 'PPO':
        agent = PPOAgent(
            actor=choose_net(env, activation='tanh'),
            critic=choose_net(env, activation='tanh', output_shape=1),
            lr=0.0005,
            gamma=0.99,
            clip_ratio=0.1,
            entropy_coeff=0.01,
        )
    else:
        logging.error('Unsupported RL algorithm specified.')

    algo = RL(
        env=env,
        agent=agent,
        use_gpu=use_gpu,
        save_path=save_path,
        load_path=load_path,
    )
    return algo


def evaluate_baselines():
    env = 'LunarLander-v2'
    algs = ['DQN', 'A2C', 'PPO']

    for alg in algs:
        logging.debug(alg)
        save_path = 'logs/%s/%s' % (alg, env)
        load_path = 'model_zoo/%s/%s' % (alg, env)

        # Build experiment -----
        algo = build(
            rl_algorithm=alg,
            save_path=save_path,
            load_path=load_path,
            env_id=env,
            num_env=16,
            use_gpu=True,
        )

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

    # LunarLander-v2 -----------------------------------------------------------

    # Build DQN experiment ----------------------------

    if not os.path.exists('logs/DQN'):
        os.makedirs('logs/DQN')

    save_path = 'examples/logs/DQN/LunarLander-v2'
    load_path = ''

    algo = build(
        rl_algorithm='DQN',
        save_path=save_path,
        load_path=load_path,
        env_id='LunarLander-v2',
        num_env=16,
        use_gpu=True,
    )

    # Build A2C experiment ----------------------------

    # if not os.path.exists('logs/A2C'):
    #     os.makedirs('logs/A2C')

    # save_path = 'examples/logs/A2C/LunarLander-v2'
    # load_path = ''

    # algo = build(
    #     rl_algorithm='A2C',
    #     save_path=save_path,
    #     load_path=load_path,
    #     env_id='LunarLander-v2',
    #     num_env=16,
    #     use_gpu=True,
    # )

    # Build PPO experiment ----------------------------

    # if not os.path.exists('logs/PPO'):
    #     os.makedirs('logs/PPO')

    # save_path = 'examples/logs/PPO/LunarLander-v2'
    # load_path = ''

    # algo = build(
    #     rl_algorithm='PPO',
    #     save_path=save_path,
    #     load_path=load_path,
    #     env_id='LunarLander-v2',
    #     num_env=16,
    #     use_gpu=True,
    # )

    # Train
    algo.train(num_episodes=10000, rollout_steps=1)

    # Load
    algo.agent.load(save_path)

    # Test
    algo.test(num_episodes=100)
