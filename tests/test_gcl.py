import os
import pickle
import shutil
import uuid
from glob import glob

import pytest
import torch

from ilpyt.agents.gcl_agent import GCLAgent
from ilpyt.agents.ppo_agent import PPOAgent
from ilpyt.algos.gcl import GCL
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


@pytest.fixture(params=[0.99])
def gamma(request):
    return request.param


@pytest.fixture(params=[0.01])
def entropy_coeff(request):
    return request.param


@pytest.fixture(params=[0.2])
def clip_ratio(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_gcl(env_id, num_env, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_env, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_gcl(env_gcl, learning_rate, gamma, clip_ratio, entropy_coeff):
    # Build agent
    actor = PPOAgent(
        actor=choose_net(env_gcl, activation='tanh'),
        critic=choose_net(env_gcl, activation='tanh', output_shape=1),
        lr=learning_rate,
        gamma=gamma,
        clip_ratio=clip_ratio,
        entropy_coeff=entropy_coeff,
    )
    agent = GCLAgent(
        actor=actor,
        cost=choose_net(
            env_gcl, output_shape=1, activation='relu', with_action_shape=True
        ),
        lr=learning_rate / 2,
        lcr_reg_cost=False,
        mono_reg_cost=False,
    )
    return agent


@pytest.fixture
def algo_gcl(temp_directory, env_id, env_gcl, agent_gcl, use_gpu):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'gcl', uuid.uuid4().hex)
    )

    # Build algorithm
    return GCL(
        env=env_gcl,
        agent=agent_gcl,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
    )


@pytest.fixture
def demos_gcl(env_id, env_gcl, temp_directory):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s.pkl' % (env_id, 'gcl', uuid.uuid4().hex)
    )

    # Create random vectors for the experience
    n = (100,)
    exp = Experiences()
    obs_shape = n + env_gcl.observation_shape
    if env_gcl.type == 'discrete':
        act_shape = n
    else:
        act_shape = n + env_gcl.action_shape
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
def train_gcl(algo_gcl, num_train_episodes, demos_gcl):
    # Train algorithm
    algo_gcl.train(
        num_episodes=num_train_episodes,
        expert_demos=demos_gcl,
    )

    # [ASSERT] Log file generated
    log_files = glob(os.path.join(algo_gcl.save_path, 'events.out.tfevents.*'))
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_gcl.agent.nets.keys():
        weight_files = glob(os.path.join(algo_gcl.save_path, name + "*.pth"))
        assert len(weight_files) <= 3

    return algo_gcl


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_gcl(
    train_gcl, env_gcl, agent_gcl, use_gpu, num_test_episodes
):
    # Build algorithm (using existing weights)
    algo_trained = GCL(
        env=env_gcl,
        agent=agent_gcl,
        use_gpu=use_gpu,
        save_path=train_gcl.save_path,
        load_path=train_gcl.save_path,
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
def test_gcl_train_only(train_gcl):
    # Clean-up
    shutil.rmtree(train_gcl.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_gcl_train_and_validate(train_and_validate_gcl):
    # Clean-up
    shutil.rmtree(train_and_validate_gcl.save_path)
