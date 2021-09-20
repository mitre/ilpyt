import os
import shutil
import uuid
from glob import glob

import pytest

from ilpyt.agents.heuristic_agent import (
    LunarLanderContinuousHeuristicAgent,
    LunarLanderHeuristicAgent,
)
from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.dagger import DAgger
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


@pytest.fixture(params=[5])
def num_epochs(request):
    return request.param


@pytest.fixture(params=[1e4])
def max_mem(request):
    return request.param


@pytest.fixture(params=[20])
def t_steps(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_dagger(env_id, num_env, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_env, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_dagger(env_dagger, learning_rate):
    # Build agent
    return ImitationAgent(net=choose_net(env_dagger), lr=learning_rate)


@pytest.fixture
def expert_dagger(env_id):
    if env_id == 'LunarLander-v2':
        return LunarLanderHeuristicAgent()
    else:
        return LunarLanderContinuousHeuristicAgent()


@pytest.fixture
def algo_dagger(
    temp_directory, env_id, env_dagger, agent_dagger, use_gpu, max_mem
):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'dagger', uuid.uuid4().hex)
    )

    # Build algorithm
    return DAgger(
        env=env_dagger,
        agent=agent_dagger,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
        max_mem=max_mem,
    )


##############################################
######### Training (Core) Function ###########
##############################################


@pytest.fixture
def train_dagger(
    algo_dagger,
    num_train_episodes,
    num_epochs,
    batch_size,
    expert_dagger,
    t_steps,
):
    # Train algorithm
    algo_dagger.train(
        num_episodes=num_train_episodes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        expert=expert_dagger,
        T_steps=t_steps,
    )

    # [ASSERT] Log file generated
    log_files = glob(
        os.path.join(algo_dagger.save_path, 'events.out.tfevents.*')
    )
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_dagger.agent.nets.keys():
        weight_files = glob(
            os.path.join(algo_dagger.save_path, name + "*.pth")
        )
        assert len(weight_files) <= 3

    return algo_dagger


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_dagger(
    train_dagger, env_dagger, agent_dagger, use_gpu, num_test_episodes, max_mem
):
    # Build algorithm (using existing weights)
    algo_trained = DAgger(
        env=env_dagger,
        agent=agent_dagger,
        use_gpu=use_gpu,
        save_path=train_dagger.save_path,
        load_path=train_dagger.save_path,
        max_mem=max_mem,
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
def test_dagger_train_only(train_dagger):
    # Clean-up
    shutil.rmtree(train_dagger.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_dagger_train_and_validate(train_and_validate_dagger):
    # Clean-up
    shutil.rmtree(train_and_validate_dagger.save_path)
