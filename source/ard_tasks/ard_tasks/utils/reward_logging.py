# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expose an ARD-designed reward's individual components to TensorBoard.

Eureka's central mechanism is that the LLM returns *two* things — a total reward and
a dict naming each reward component — and every component is logged during training,
so the next iteration's prompt can show the LLM what each component it wrote actually
did. Without that, feedback tips like "if a component's values are near identical, RL
cannot optimise it, so rescale or discard it" have no data to stand on.

This module is the framework half of that contract. Each task's ``_get_rewards`` calls
``compute_reward`` (the sole ARD edit target), then hands the result here:

    def _get_rewards(self) -> torch.Tensor:
        total_reward, reward_components = self.compute_reward()
        log_reward_components(self, total_reward, reward_components)
        return total_reward

Values land in ``self.extras["log"]``. IsaacLab's rl_games wrapper renames that key
``"log"`` -> ``"episode"`` in flight, and rl_games' ``IsaacAlgoObserver`` writes every
entry to TensorBoard as ``Episode/<key>``.

Three properties of that pipeline shape the code below.

1. ``self.extras`` is created once in ``DirectRLEnv.__init__`` and never cleared, and
   the wrapper pops ``"log"`` out of a *copy* of it — so the env's own ``extras["log"]``
   dict survives every step. Left alone, one dict object gets mutated in place and
   appended to the observer's ``ep_infos`` once per step, and the epoch's TensorBoard
   value collapses to the last step's reading instead of the epoch mean. Hence
   ``reset_episode_log``, which every task calls from ``_get_dones`` (the first of the
   two per-step hooks) to start each step with a fresh dict.
2. ``IsaacAlgoObserver.after_print_stats`` reads the key set from ``ep_infos[0]`` and
   indexes *every* later dict with it, so a key that appears on some steps and not
   others raises ``KeyError`` mid-training. LLM-written code is exactly where that
   happens, so the key set is pinned on first use and reconciled on every later step.
3. Everything shares one flat ``Episode/`` namespace with the fixed evaluation metric,
   so component names are prefixed (``rew_``) and can never collide with it.
"""

from __future__ import annotations

import torch

# Prefix for every scalar this module logs. Keeps ARD-designed components in their own
# TensorBoard namespace (``Episode/rew_*``), and guarantees a component can never be
# named ``fitness_function`` and shadow the metric the reward is scored on.
COMPONENT_PREFIX = "rew_"

# The aggregate, logged alongside the components (Eureka logs this as ``gpt_reward``).
# The LLM needs it to judge whether a component's magnitude is large or small relative
# to the reward as a whole — one of the feedback tips it is asked to act on.
TOTAL_KEY = COMPONENT_PREFIX + "total"

# Attribute the pinned key set is cached under on the env instance.
_KEYS_ATTR = "_ard_reward_component_keys"


def reset_episode_log(env) -> dict:
    """Start this step's ``extras["log"]`` with a fresh dict and return it.

    Call this once per step, from ``_get_dones`` (which ``DirectRLEnv.step`` runs before
    ``_get_rewards``), so the per-step dict the rl_games observer collects is a distinct
    object each step. See point 1 in the module docstring.
    """
    env.extras["log"] = dict()
    return env.extras["log"]


def log_reward_components(env, total_reward: torch.Tensor, components) -> None:
    """Write ``total_reward`` and each reward component to ``env.extras["log"]``.

    Each value is reduced to its mean over environments before logging: the observer
    concatenates every step's readings over a whole epoch, so logging full
    ``(num_envs,)`` tensors would accumulate thousands of floats per component per step.

    Args:
        env: The task environment (needs ``.extras`` and ``.device``).
        total_reward: Per-env total reward, shape ``(num_envs,)``.
        components: Mapping of component name -> per-env tensor. A non-dict (e.g. the
            ``None`` a partially written reward might return) logs the total only.
    """
    log = env.extras.setdefault("log", dict())
    log[TOTAL_KEY] = _scalar(total_reward, env)

    if not isinstance(components, dict):
        return

    values = {
        COMPONENT_PREFIX + str(name): _scalar(value, env)
        for name, value in components.items()
    }

    # Pin the key set on first use, then force every later step to carry exactly those
    # keys: a missing one is logged as 0.0, an unexpected one is dropped. Without this a
    # conditionally built component dict kills the run partway through. See point 2.
    keys = getattr(env, _KEYS_ATTR, None)
    if keys is None:
        keys = tuple(values)
        setattr(env, _KEYS_ATTR, keys)

    zero = torch.zeros((), device=env.device)
    for key in keys:
        log[key] = values.get(key, zero)


def _scalar(value, env) -> torch.Tensor:
    """Reduce ``value`` to a 0-d tensor on the env device, tolerating a plain number."""
    if not isinstance(value, torch.Tensor):
        return torch.tensor(float(value), device=env.device)
    return value.float().mean()
