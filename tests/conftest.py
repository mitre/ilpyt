import tempfile

import pytest

##############################################
##### Fixture (Session-Based) Parameters #####
##############################################


@pytest.fixture(scope="session", params=[False, True])
def use_gpu(request):
    return request.param


@pytest.fixture(scope="session", params=[4])
def num_env(request):
    return request.param


@pytest.fixture(scope="session", params=[24])
def seed(request):
    return request.param


@pytest.fixture(scope="session", params=['dummy'])
def vecenv_type(request):
    return request.param


@pytest.fixture(scope="session")
def temp_directory():
    return tempfile.mkdtemp()


@pytest.fixture(params=[5])
def num_train_episodes(request):
    return request.param


@pytest.fixture(params=[5])
def num_test_episodes(request):
    return request.param
