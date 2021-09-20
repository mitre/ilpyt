import os
import pickle
import shutil
import uuid
from glob import glob

import pytest
import torch

from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.bc import BC
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.runners.runner import Experiences
from ilpyt.utils.seed_utils import set_seed

##############################################
#### Fixture (Function-Based) Parameters #####
##############################################


@pytest.fixture(
    params=[
        'LunarLander-v2',
        'LunarLanderContinuous-v2',
        'PongDeterministic-v4',
        'CarRacing-v0',
    ]
)
def env_id(request):
    return request.param


@pytest.fixture(params=[0.001])
def learning_rate(request):
    return request.param


@pytest.fixture(params=[10])
def batch_size(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_bc(env_id, num_env, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_env, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_bc(env_bc, learning_rate):
    # Build agent
    return ImitationAgent(net=choose_net(env_bc), lr=learning_rate)


@pytest.fixture
def algo_bc(temp_directory, env_id, env_bc, agent_bc, use_gpu):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'bc', uuid.uuid4().hex)
    )

    # Build algorithm
    return BC(
        env=env_bc,
        agent=agent_bc,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
    )


@pytest.fixture
def demos_bc(env_id, env_bc, temp_directory):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s.pkl' % (env_id, 'bc', uuid.uuid4().hex)
    )

    # Create random vectors for the experience
    n = (100,)
    exp = Experiences()
    obs_shape = n + env_bc.observation_shape
    if env_bc.type == 'discrete':
        act_shape = n
    else:
        act_shape = n + env_bc.action_shape
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
def train_bc(algo_bc, num_train_episodes, batch_size, demos_bc):
    # Train algorithm
    algo_bc.train(
        num_epochs=num_train_episodes,
        batch_size=batch_size,
        expert_demos=demos_bc,
    )

    # [ASSERT] Log file generated
    log_files = glob(os.path.join(algo_bc.save_path, 'events.out.tfevents.*'))
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_bc.agent.nets.keys():
        weight_files = glob(os.path.join(algo_bc.save_path, name + "*.pth"))
        assert len(weight_files) <= 3

    return algo_bc


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_bc(
    train_bc, env_bc, agent_bc, use_gpu, num_test_episodes
):
    # Build algorithm (using existing weights)
    algo_trained = BC(
        env=env_bc,
        agent=agent_bc,
        use_gpu=use_gpu,
        save_path=train_bc.save_path,
        load_path=train_bc.save_path,
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
def test_bc_train_only(train_bc):
    # Clean-up
    shutil.rmtree(train_bc.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_bc_train_and_validate(train_and_validate_bc):
    # Clean-up
    shutil.rmtree(train_and_validate_bc.save_path)
