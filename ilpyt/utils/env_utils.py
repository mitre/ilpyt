import gym

from ilpyt.envs.dummy_vec_env import DummyVecEnv
from ilpyt.envs.subproc_vec_env import SubprocVecEnv
from ilpyt.envs.vec_env import VecEnv

import gym

from ilpyt.envs.dummy_vec_env import DummyVecEnv
from ilpyt.envs.subproc_vec_env import SubprocVecEnv
from ilpyt.envs.vec_env import VecEnv


def build_env(
    env_id: str, num_env: int = 16, seed: int = 24, vecenv_type: str = "dummy"
) -> VecEnv:
    """
    Build vectorized environment. Adapted from: https://github.com/openai/baselines/

    Parameters
    ----------
    env_id: str
        Name of registered OpenAI Gym environment
    num_env: int, default=16
        number of parallel environments to initialize
    seed: int, default=24
        random seed for environments
    vecenv_type: str, default="dummy"
        Vectorized environment type; choose from ["dummy", "subproc"]; 
        "dummy" will create a serialized environment, which is useful for debugging;
        "subproc" will create parallel environments for high-throughput training

    Returns
    -------
    VecEnv:
        vectorized environment
    """

    def make_env(env_id, rank):
        env = gym.make(env_id)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        return env

    if vecenv_type == 'dummy':
        vec_env = DummyVecEnv(
            [lambda rank=i: make_env(env_id, rank) for i in range(num_env)]
        )  # type: VecEnv

    elif vecenv_type == 'subproc':
        vec_env = SubprocVecEnv(
            [lambda rank=i: make_env(env_id, rank) for i in range(num_env)]
        )

    return vec_env


def list_all_envs() -> None:
    """
    List all the environments available on this system.
    Please see the official list of OpenAI Gym environments for more details: https://github.com/openai/gym/wiki/Table-of-environments
    """
    env_ids = [env_spec.id for env_spec in gym.envs.registry.all()]
    for env_id in env_ids:
        try:
            env = gym.make(env_id)
            print(env_id, env.observation_space, env.action_space)
        except Exception:  # Mujoco
            print(env_id, "not available on this system.")
