import os
import shutil
import uuid
from glob import glob

import pytest

from ilpyt.agents.a2c_agent import A2CAgent
from ilpyt.algos.rl import RL
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
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


@pytest.fixture(params=[0.1])
def entropy_coeff(request):
    return request.param


@pytest.fixture(params=[16])
def rollout_steps(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_a2c(env_id, num_env, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_env, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_a2c(env_a2c, learning_rate, gamma, entropy_coeff):
    # Build agent
    return A2CAgent(
        actor=choose_net(env_a2c),
        critic=choose_net(env_a2c, output_shape=1),
        lr=learning_rate,
        gamma=gamma,
        entropy_coeff=entropy_coeff,
    )


@pytest.fixture
def algo_a2c(temp_directory, env_id, env_a2c, agent_a2c, use_gpu):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'a2c', uuid.uuid4().hex)
    )

    # Build algorithm
    return RL(
        env=env_a2c,
        agent=agent_a2c,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
    )


##############################################
######### Training (Core) Function ###########
##############################################


@pytest.fixture
def train_a2c(algo_a2c, num_train_episodes, rollout_steps):
    # Train algorithm
    algo_a2c.train(
        num_episodes=num_train_episodes, rollout_steps=rollout_steps
    )

    # [ASSERT] Log file generated
    log_files = glob(os.path.join(algo_a2c.save_path, 'events.out.tfevents.*'))
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_a2c.agent.nets.keys():
        weight_files = glob(os.path.join(algo_a2c.save_path, name + "*.pth"))
        assert len(weight_files) <= 3

    return algo_a2c


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_a2c(
    train_a2c, env_a2c, agent_a2c, use_gpu, num_test_episodes, rollout_steps
):
    # Build algorithm (using existing weights)
    algo_trained = RL(
        env=env_a2c,
        agent=agent_a2c,
        use_gpu=use_gpu,
        save_path=train_a2c.save_path,
        load_path=train_a2c.save_path,
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
def test_a2c_train_only(train_a2c):
    # Clean-up
    shutil.rmtree(train_a2c.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_a2c_train_and_validate(train_and_validate_a2c):
    # Clean-up
    shutil.rmtree(train_and_validate_a2c.save_path)
