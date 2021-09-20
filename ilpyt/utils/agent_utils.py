from typing import Any, Dict, Tuple

import torch


def soft_update(
    source_net: torch.nn.Module, target_net: torch.nn.Module, tau: float
) -> None:
    """
    Mix network parameters from `source_net` to `target_net` with mixing factor 
    `tau`.

    The target network parameters will be:
        target = tau * source + (1-tau) * target

    Parameters
    ----------
    source_net: torch.nn.Module:
        network with source weights
    target_net: torch.nn.Module:
        target network for source weights
    tau: float
        mixing factor between source and target
    """
    for p0, p1 in zip(source_net.parameters(), target_net.parameters()):
        p1.data.copy_(tau * p0.data + (1.0 - tau) * p1.data)


def hard_update(
    source_net: torch.nn.Module, target_net: torch.nn.Module
) -> None:
    """
    Copy network parameters from `source_net` to `target_net`.

    Parameters
    ----------
    source_net: torch.nn.Module:
        network with source weights
    target_net: torch.nn.Module:
        target network for source weights
    """
    for p0, p1 in zip(source_net.parameters(), target_net.parameters()):
        p1.data.copy_(p0.data)


def flatten_batch(batch: Dict) -> Dict:
    """
    Flatten batch of rollouts  with the shape (rollout_steps, num_env,
    item_shape) to (rollout_steps * num_env, item_shape).
        [t0/env0, t0/env1, ..., t1/env0, t1/env1, ..., t2/env0, t2/env1, ..]

    Parameters
    ----------
    batch: dict[str, torch.Tensor]
        Batch of rollouts

    Returns
    -------
    dict[str, torch.Tensor]: flattened batch of rollouts
    """
    for key, value in batch.items():
        if key == 'infos':
            continue
        batch[key] = flatten_tensor(value)
    return batch


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten tensor from shape (rollout_steps, num_env, item_shape) to 
    (rollout_steps * num_env, item_shape). A helper function for `flatten_batch`.

    Parameters
    ----------
    x: torch.Tensor
        input tensor of shape (rollout_steps, num_env, item_shape)

    Returns
    -------
    torch.Tensor:
        output tensor of shape (rollout_step*num_env, item_shape)
    """
    if len(x.shape) <= 1:
        return x
    rollout_steps, num_env = x.shape[:2]
    new_shape = (rollout_steps * num_env,)  # type: Tuple[Any]
    if len(x.shape) > 2:
        new_shape += tuple(x.shape[2:])
    return x.reshape(new_shape)


def compute_target(
    value_final: torch.Tensor,
    rewards: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Compute target (sum of total discounted rewards) for rollout.

    Parameters
    -----------
    value_final: torch.Tensor
        state values from final time step of rollout, size (num_env,)
    rewards: torch.Tensor
        rewards across rollout, size (rollout_steps, num_env)
    masks: torch.Tensor
        masks for episode end states, 0 if end state, 1 otherwise,
        size (rollout_steps, num_env)
    gamma: float
        discount factor for rollout

    Returns
    -------
    torch.Tensor: targets, size (rollout_steps, num_env)
    """
    G = value_final
    T = rewards.shape[0]
    targets = torch.zeros(rewards.shape)

    for i in range(T - 1, -1, -1):
        G = rewards[i] + gamma * G * masks[i]
        targets[i] = G

    return targets
