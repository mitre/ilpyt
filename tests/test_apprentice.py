import os
import pickle
import shutil
import uuid
from glob import glob

import pytest
import torch

from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.algos.apprentice import Apprentice
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.runners.runner import Experiences
from ilpyt.utils.seed_utils import set_seed

##############################################
#### Fixture (Function-Based) Parameters #####
##############################################


@pytest.fixture(
    params=[
        'CartPole-v0',
        'MountainCar-v0',
        'LunarLander-v2',
    ]
)
def env_id(request):
    return request.param


@pytest.fixture(params=[0.001])
def learning_rate(request):
    return request.param


@pytest.fixture(params=[0.9])
def gamma(request):
    return request.param


@pytest.fixture(params=[0.0001])
def epsilon(request):
    return request.param


@pytest.fixture(params=[10])
def num_train(request):
    return request.param


@pytest.fixture(params=[64])
def batch_size(request):
    return request.param


@pytest.fixture(params=[10])
def replay_memory_size(request):
    return request.param


@pytest.fixture(params=[0.9])
def epsilon_start(request):
    return request.param


@pytest.fixture(params=[0.01])
def epsilon_end(request):
    return request.param


@pytest.fixture(params=[1e5])
def epsilon_steps(request):
    return request.param


@pytest.fixture(params=[0.05])
def tau(request):
    return request.param


@pytest.fixture(params=[16])
def num_envs(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_app(env_id, num_envs, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_envs, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_app(
    env_app,
    learning_rate,
    replay_memory_size,
    epsilon_start,
    epsilon_end,
    epsilon_steps,
    tau,
    gamma,
    batch_size,
    num_envs,
):
    # Build agent
    return DQNAgent(
        net=choose_net(env_app),
        target_net=choose_net(env_app),
        num_actions=env_app.num_actions,
        lr=learning_rate,
        replay_memory_size=replay_memory_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_steps=epsilon_steps,
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        num_envs=num_envs,
    )


@pytest.fixture
def algo_app(temp_directory, env_id, env_app, agent_app, use_gpu):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'apprentice', uuid.uuid4().hex)
    )

    # Build algorithm
    return Apprentice(
        env=env_app,
        agent=agent_app,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
    )


@pytest.fixture
def demos_app(env_id, env_app, temp_directory):
    store_path = os.path.join(
        temp_directory,
        '%s_%s_%s.pkl' % (env_id, 'apprentice', uuid.uuid4().hex),
    )

    # Create random vectors for the experience
    n = (100,)
    exp = Experiences()
    obs_shape = n + env_app.observation_shape
    if env_app.type == 'discrete':
        act_shape = n
    else:
        act_shape = n + env_app.action_shape
    exp.states = torch.zeros(obs_shape)
    exp.actions = torch.zeros(act_shape)
    exp.rewards = torch.zeros(n)
    exp.dones = torch.zeros(n)

    # Save to pickle
    with open(store_path, 'wb') as f:
        pickle.dump(exp, f, pickle.HIGHEST_PROTOCOL)

    return store_path


##############################################
######### Training (Core) Function ###########
##############################################


@pytest.fixture
def train_app(
    algo_app,
    num_train_episodes,
    num_train,
    gamma,
    epsilon,
    batch_size,
    demos_app,
):
    # Train algorithm
    algo_app.train(
        num_episodes=num_train_episodes,
        num_train=num_train,
        batch_size=batch_size,
        gamma=gamma,
        epsilon=epsilon,
        expert_demos=demos_app,
    )

    # [ASSERT] Log file generated
    log_files = glob(os.path.join(algo_app.save_path, 'events.out.tfevents.*'))
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_app.agent.nets.keys():
        weight_files = glob(os.path.join(algo_app.save_path, name + "*.pth"))
        assert len(weight_files) == num_train_episodes

    return algo_app


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_app(
    train_app, env_app, agent_app, use_gpu, num_test_episodes
):
    # Build algorithm (using existing weights)
    algo_trained = Apprentice(
        env=env_app,
        agent=agent_app,
        use_gpu=use_gpu,
        save_path=train_app.save_path,
        load_path=train_app.save_path,
    )

    # Test algorithm
    algo_trained.test(num_episodes=num_test_episodes)

    # [ASSERT] Results file generated
    results_file = os.path.join(algo_trained.save_path, 'test_results.npy')
    assert os.path.exists(results_file)

    return algo_trained


##############################################
########## Training Only (PyTest) ############
##############################################


@pytest.mark.skip
def test_app_train_only(train_app):
    # Clean-up
    shutil.rmtree(train_app.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_app_train_and_validate(train_and_validate_app):
    # Clean-up
    shutil.rmtree(train_and_validate_app.save_path)
