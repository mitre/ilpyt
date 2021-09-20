import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets random seed across `random`, `numpy`, and `torch` for experiment 
    replicability.

    Parameters
    ----------
    seed: int
        seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
