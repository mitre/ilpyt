import os
import shutil
import uuid
from glob import glob

import pytest

from ilpyt.agents.dqn_agent import DQNAgent
from ilpyt.algos.rl import RL
from ilpyt.utils.env_utils import build_env
from ilpyt.utils.net_utils import choose_net
from ilpyt.utils.seed_utils import set_seed

##############################################
#### Fixture (Function-Based) Parameters #####
##############################################


@pytest.fixture(params=['LunarLander-v2', 'PongDeterministic-v4'])
def env_id(request):
    return request.param


@pytest.fixture(params=[0.001])
def learning_rate(request):
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


@pytest.fixture(params=[0.99])
def gamma(request):
    return request.param


@pytest.fixture(params=[64])
def batch_size(request):
    return request.param


@pytest.fixture(params=[16])
def rollout_steps(request):
    return request.param


@pytest.fixture(params=[16])
def num_envs(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_dqn(env_id, num_envs, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_envs, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_dqn(
    env_dqn,
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
        net=choose_net(env_dqn),
        target_net=choose_net(env_dqn),
        num_actions=env_dqn.num_actions,
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
def algo_dqn(temp_directory, env_id, env_dqn, agent_dqn, use_gpu):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'dqn', uuid.uuid4().hex)
    )

    # Build algorithm
    return RL(
        env=env_dqn,
        agent=agent_dqn,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
    )


##############################################
######### Training (Core) Function ###########
##############################################


@pytest.fixture
def train_dqn(algo_dqn, num_train_episodes, rollout_steps):
    # Train algorithm
    algo_dqn.train(
        num_episodes=num_train_episodes, rollout_steps=rollout_steps
    )

    # [ASSERT] Log file generated
    log_files = glob(os.path.join(algo_dqn.save_path, 'events.out.tfevents.*'))
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_dqn.agent.nets.keys():
        weight_files = glob(os.path.join(algo_dqn.save_path, name + "*.pth"))
        assert len(weight_files) <= 3

    return algo_dqn


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_dqn(
    train_dqn, env_dqn, agent_dqn, use_gpu, num_test_episodes, rollout_steps
):
    # Build algorithm (using existing weights)
    algo_trained = RL(
        env=env_dqn,
        agent=agent_dqn,
        use_gpu=use_gpu,
        save_path=train_dqn.save_path,
        load_path=train_dqn.save_path,
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
def test_dqn_train_only(train_dqn):
    # Clean-up
    shutil.rmtree(train_dqn.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_dqn_train_and_validate(train_and_validate_dqn):
    # Clean-up
    shutil.rmtree(train_and_validate_dqn.save_path)
