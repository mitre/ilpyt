import os
import shutil
import uuid
from glob import glob

import pytest

from ilpyt.agents.ppo_agent import PPOAgent
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
        'CarRacing-v0',
        'PongDeterministic-v4',
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


@pytest.fixture(params=[0.2])
def clip_ratio(request):
    return request.param


@pytest.fixture(params=[5])
def num_train_episodes(request):
    return request.param


@pytest.fixture(params=[16])
def rollout_steps(request):
    return request.param


##############################################
############## Setup Functions ###############
##############################################


@pytest.fixture
def env_ppo(env_id, num_env, seed, vecenv_type):
    # Set random seed
    set_seed(seed)

    # Build environment
    return build_env(
        env_id=env_id, num_env=num_env, seed=seed, vecenv_type=vecenv_type
    )


@pytest.fixture
def agent_ppo(env_ppo, learning_rate, gamma, clip_ratio, entropy_coeff):
    # Build agent
    return PPOAgent(
        actor=choose_net(env_ppo),
        critic=choose_net(env_ppo, output_shape=1),
        lr=learning_rate,
        gamma=gamma,
        clip_ratio=clip_ratio,
        entropy_coeff=entropy_coeff,
    )


@pytest.fixture
def algo_ppo(temp_directory, env_id, env_ppo, agent_ppo, use_gpu):
    store_path = os.path.join(
        temp_directory, '%s_%s_%s' % (env_id, 'ppo', uuid.uuid4().hex)
    )

    # Build algorithm
    return RL(
        env=env_ppo,
        agent=agent_ppo,
        use_gpu=use_gpu,
        save_path=store_path,
        load_path='',
    )


##############################################
######### Training (Core) Function ###########
##############################################


@pytest.fixture
def train_ppo(algo_ppo, num_train_episodes, rollout_steps):
    # Train algorithm
    algo_ppo.train(
        num_episodes=num_train_episodes, rollout_steps=rollout_steps
    )

    # [ASSERT] Log file generated
    log_files = glob(os.path.join(algo_ppo.save_path, 'events.out.tfevents.*'))
    assert len(log_files) == 1

    # [ASSERT] Weights file generated
    for name in algo_ppo.agent.nets.keys():
        weight_files = glob(os.path.join(algo_ppo.save_path, name + "*.pth"))
        assert len(weight_files) <= 3

    return algo_ppo


##############################################
######## Validation (Core) Function ##########
##############################################


@pytest.fixture
def train_and_validate_ppo(
    train_ppo, env_ppo, agent_ppo, use_gpu, num_test_episodes, rollout_steps
):
    # Build algorithm (using existing weights)
    algo_trained = RL(
        env=env_ppo,
        agent=agent_ppo,
        use_gpu=use_gpu,
        save_path=train_ppo.save_path,
        load_path=train_ppo.save_path,
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
def test_ppo_train_only(train_ppo):
    # Clean-up
    shutil.rmtree(train_ppo.save_path)


##############################################
######## Training & Testing (PyTest) #########
##############################################

# @pytest.mark.skip
def test_ppo_train_and_validate(train_and_validate_ppo):
    # Clean-up
    shutil.rmtree(train_and_validate_ppo.save_path)
