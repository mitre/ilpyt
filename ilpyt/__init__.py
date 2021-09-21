"""
# ilpyt

The imitation learning toolbox (`ilpyt`) contains modular implementations of 
common deep imitation learning algorithms in PyTorch, with unified 
infrastructure supporting key imitation learning and reinforcement learning 
algorithms. You can read more about `ilpyt` in our 
[white paper](https://github.com/mitre/ilpyt/blob/main/docs/ilpyt_white_paper.pdf).

Documentation is available [here](https://mitre.github.io/ilpyt).

## Table of Contents
- [Main Features](#main-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Getting Started: Basic Usage](#basic-usage)
- [Getting Started: Code Organization](#code-organization) 
- [Getting Started: Customization](#customization)
- [Supported Algorithms and Environments](#supported-algorithms-and-environments)
- [Benchmarks](#benchmarks)
- [Citation](#citation)

## Main Features

- Implementation of baseline imitation learning algorithms: BC, DAgger, Apprenticeship learning, GCL, GAIL.
- Implementation of baseline reinforcement learning algorithms, for comparison purposes: DQN, A2C, PPO.
- Modular, extensible framework for training, evaluating, and testing imitation learning (and reinforcement learning) algorithms.
- Simple algorithm API which exposes train and test methods, allowing for quick library setup and use (a basic usage of the library requires less than ten lines of code to have a fully functioning train and test pipeline).
- A modular infrastructure for easy modification and reuse of existing components for novel algorithm implementations.
- Parallel and serialization modes, allowing for faster, optimized operations or serial operations for debugging.
- Compatibility with the OpenAI Gym environment interface for access to many existing benchmark learning environments, as well as the flexibility to create custom environments.

## Installation

Note: `ilpyt` has only been tested on Ubuntu 20.04, and with Python 3.8.5. 

**STEP 1** 

In order to install `ilpyt`, there are a few prerequisites required. The following commands will setup all the basics so you can run `ilpyt` with the OpenAI Gym environments:

```
# Install system-based packages
apt-get install cmake python3-pip python3-testresources freeglut3-dev xvfb

# Install Wheel
pip3 install --no-cache-dir --no-warn-script-location wheel
```

**STEP 2** 

Install `ilpyt` using pip:

```
pip3 install ilpyt

# Or to install from source:
# pip3 install -e .
```
**STEP 3** (Optional) 

Run the associated Python tests to confirm the package has 
installed successfully: 

```
git clone https://github.com/mitre/ilpyt.git
cd ilpyt/

# To run all the tests
# If running headless, prepend the pytest command with `xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" --`
pytest tests/

# Example: to run an individual test, like DQN
pytest tests/test_dqn.py 
```

## Getting Started

Various sample Python script(s) of how to run the toolbox can be found within 
the `examples` directory. Documentation is available 
[here](https://mitre.github.io/ilpyt).

### Basic Usage

Various sample Python script(s) of how to run the toolbox can be found within 
the `examples` directory. A minimal train and test snippet for an imitation 
learning algorithm takes less than 10 lines of code in `ilpyt`. In this basic 
example, we are training a behavioral cloning algorithm for 10,000 epochs before 
testing the best policy for 100 episodes.

```py
import ilpyt
from ilpyt.agents.imitation_agent import ImitationAgent
from ilpyt.algos.bc import BC

env = ilpyt.envs.build_env(env_id='LunarLander-v2',  num_env=16)
net = ilpyt.nets.choose_net(env)
agent = ImitationAgent(net=net, lr=0.0001)

algo = BC(agent=agent, env=env)
algo.train(num_epochs=10000, expert_demos='demos/LunarLander-v2/demos.pkl')
algo.test(num_episodes=100)
```

### Code Organization 

The main components of the `ilpyt` library are the `algorithm`, `environment`, 
`agent`, `net`, and `runner`. Please see our 
[conceptual structure diagram](https://github.com/mitre/ilpyt/docs/figures/conceptual_structure.png) 
for more details.

At a high-level, the `algorithm` orchestrates the training and testing of our 
`agent` in a particular `environment`. During these training or testing loops, 
a `runner` will execute the `agent` and `environment` in a loop to collect 
(`state`, `action`, `reward`, `next state`) transitions. The individual 
components of a transition (e.g., `state` or `action`) are typically torch 
Tensors. The `agent` can then use this batch of transitions to update its 
`network` and move towards an optimal action policy.

### Customization

To implement a new algorithm, one simply has to extend the BaseAlgorithm and 
BaseAgent abstract classes (for even further customization, one can even make 
custom networks by extending the BaseNetwork interface). Each of these 
components is modular (see [code organization](#code-organization) for more 
details), allowing components to be easily swapped out. (For example, the 
agent.generator used in the GAIL algorithm can be easily swapped between 
PPOAgent, DQNAgent, or A2Cagent. In a similar way, new algorithm implementations 
can utilize existing implemented classes as building blocks, or extend the class 
interfaces for more customization.)

Adding a custom environment is as simple as extending the OpenAI Gym Environment 
interface and registering it within your local gym environment registry.

See `agents/base_agent.py`, `algos/base_algo.py`, `nets/base_net.py` for more 
details.

## Supported Algorithms and Environments

The following imitation learning (IL) algorithms are supported:

* [Behavioral Cloning](https://papers.nips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf) (BC)
* [Dataset Aggregation](https://arxiv.org/pdf/1011.0686.pdf) (DAgger)
* [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (GAIL)
* [Apprenticeship Learning](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf) (AppL)
* [Guided Cost Learning](https://arxiv.org/abs/1603.00448) (GCL)

The following reinforcement learning (RL) algorithms are supported:

* [Advantage Actor Critic](https://arxiv.org/abs/1602.01783) (A2C)
* [Deep Q-Networks](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) (DQN)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)

The following [OpenAI Gym Environments](https://github.com/openai/gym/wiki/Table-of-environments) are supported. Environments with:

* Observation space: `Box(x,)` and `Box(x,y,z)`
* Action space: `Discrete(x)` and `Box(x,)`

NOTE: To create your own custom environment, just follow the OpenAI Gym 
Environment interface. i.e., your environment must implement the following 
methods (and inherit from the OpenAI Gym Class). More detailed instructions can 
be found on the OpenAI GitHub repository page on creating 
[custom Gym environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md).

## Benchmarks

Sample train and test results of the baseline algorithms on some environments:

| |CartPole-v0 | MountainCar-v0 | MountainCarContinuous-v0 | LunarLander-v2 | LunarLanderContinuous-v2 |
| -- | -- | -- | -- | -- | -- | 
| Threshold | 200 | -110  |  90 | 200 |  200 | 
Expert (Mean/Std) |  200.00 / 0.00 | -98.71 / 7.83 |  93.36 / 0.05 | 268.09 / 21.18 |  283.83 / 17.70 | 
BC (Mean/Std) |  200.00 / 0.00 | -100.800 / 13.797 | 93.353 / 0.113 | 244.295 / 97.765 |  285.895 / 14.584 |
DAgger (Mean/Std) | 200.00 / 0.00 | -102.36 / 15.38 |  93.20 / 0.17 |  230.15 / 122.604 | 285.85 / 14.61 | 
GAIL (Mean/Std) | 200.00 / 0.00 | -104.31 / 17.21 |  79.78 / 6.23 |  201.88 / 93.82 | 282.00 / 31.73 |
GCL | 200.00 / 0.00 | - | - | 212.321 / 119.933 | 255.414 / 76.917 |
AppL(Mean/Std) |   200.00 / 0.00 | -108.60 / 22.843 | -  | -  | -  |
DQN (Mean/Std) |  - | - |  - |  281.96 / 24.57 | - |
A2C (Mean/Std) |  - |  | - | 201.26 / 62.52 |   - |
PPO (Mean/Std) |  - | - | -  | 249.72 / 75.05 | -  |

The pre-trained weights for these models can be found in our 
[Model Zoo](https://github.com/mitre/ilpyt/tree/main/model_zoo).

## Citation

If you use `ilpyt` for your work, please cite our 
[white paper](https://github.com/mitre/ilpyt/blob/main/docs/ilpyt_white_paper.pdf):
```
@misc{ilpyt_2021,
  author = {Vu, Amanda and Tapley, Alex and Bissey, Brett},
  title = {ilpyt: Imitation Learning Research Code Base in PyTorch},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/mitre/ilpyt}},
}
```
"""
