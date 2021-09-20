"""
`VecEnv` is a vectorized OpenAI Gym environment object. Adapted from: 
https://github.com/openai/baselines/
"""

from abc import ABC, abstractmethod

import gym
import numpy as np


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image.
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    Parameters
    ----------
        img_nhwc: list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    Returns
    -------
        bigim_HWc: ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(
        list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)]
    )
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a 
    batch of actions to be applied per-environment.
    """

    closed = False
    viewer = None

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs

        # Observation space
        self.observation_space = observation_space
        self.observation_shape = observation_space.shape

        # Action space
        self.action_space = action_space
        if isinstance(self.action_space, gym.spaces.Box):
            self.num_actions = self.action_space.shape[0]
            self.action_shape = self.action_space.shape
            self.type = 'continuous'
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.num_actions = self.action_space.n
            self.action_shape = (1,)
            self.type = 'discrete'

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        return bigimg

    def get_images(self):
        """
        Return RGB images from each environment.
        """
        raise NotImplementedError
