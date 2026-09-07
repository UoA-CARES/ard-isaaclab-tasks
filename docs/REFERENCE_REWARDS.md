# Reference rewards (official IsaacLab implementations)

This file is the **archive** of the reward functions that used to live inside each
task's `_get_rewards`. They were removed from the env files so ARD starts from a
clean slate: the LLM never sees a reference reward, and no reference reward leaks
into its prompt. Nothing imports this file — it is documentation only.

Each task's editable reward now lives in `compute_reward` (see
[the ARD reward contract](#the-ard-reward-contract) at the end), which ships blank.
Use the code below to restore a task's official reward by hand when you need a
baseline to compare an ARD-designed reward against.

Source: IsaacLab 2.3.2, the `Isaac-Cartpole-Direct-v0`,
`Isaac-Repose-Cube-Shadow-Direct-v0`, and `Isaac-Repose-Cube-Shadow-Vision-Direct-v0`
environments. Reproduced verbatim apart from the `self.*` reads that the
ard-isaaclab-tasks migration already hoisted out of the reward (see the notes).

---

## Cartpole — `Isaac-ARD-Cartpole-v0`

`source/ard_tasks/ard_tasks/tasks/direct/cartpole/cartpole_env.py`

Reward scales, from `CartpoleEnvCfg`:

| cfg field | value |
|---|---|
| `rew_scale_alive` | `1.0` |
| `rew_scale_terminated` | `-2.0` |
| `rew_scale_pole_pos` | `-1.0` |
| `rew_scale_cart_vel` | `-0.01` |
| `rew_scale_pole_vel` | `-0.005` |

```python
def compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
    pole_vel = self.joint_vel[:, self._pole_dof_idx[0]]
    cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]

    rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
    rew_termination = self.cfg.rew_scale_terminated * self.reset_terminated.float()
    rew_pole_pos = self.cfg.rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = self.cfg.rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = self.cfg.rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel

    return total_reward, {
        "alive": rew_alive,
        "termination": rew_termination,
        "pole_pos": rew_pole_pos,
        "cart_vel": rew_cart_vel,
        "pole_vel": rew_pole_vel,
    }
```

The official IsaacLab source computes this in a `@torch.jit.script` free function
`compute_rewards(...)` taking the scales and joint states as arguments; the
migration inlined it and read the same quantities off `self`. The component dict
is new (the official function returned only the total).

---

## Shadow Hand repose (state) — `Isaac-ARD-Repose-Cube-Shadow-Direct-v0`

`source/ard_tasks/ard_tasks/tasks/direct/shadow_hand/shadow_hand_env.py`

Reward scales, from `ShadowHandEnvCfg`:

| cfg field | value |
|---|---|
| `dist_reward_scale` | `-10.0` |
| `rot_reward_scale` | `1.0` |
| `rot_eps` | `0.1` |
| `action_penalty_scale` | `-0.0002` |
| `reach_goal_bonus` | `250` |
| `fall_penalty` | `0` |
| `fall_dist` | `0.24` |
| `success_tolerance` | `0.1` |

```python
def compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dist_rew = self.goal_dist * self.cfg.dist_reward_scale
    rot_rew = 1.0 / (torch.abs(self.rot_dist) + self.cfg.rot_eps) * self.cfg.rot_reward_scale
    action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale

    total_reward = dist_rew + rot_rew + action_penalty
    # success bonus: object orientation within `success_tolerance` of the goal this step
    goal_bonus = torch.where(
        self.goal_resets.bool(),
        torch.full_like(total_reward, float(self.cfg.reach_goal_bonus)),
        torch.zeros_like(total_reward),
    )
    # fall penalty: object drifted past `fall_dist` from the in-hand position
    fall_pen = torch.where(
        self.goal_dist >= self.cfg.fall_dist,
        torch.full_like(total_reward, float(self.cfg.fall_penalty)),
        torch.zeros_like(total_reward),
    )
    total_reward = total_reward + goal_bonus + fall_pen

    return total_reward, {
        "dist": dist_rew,
        "rot": rot_rew,
        "action_penalty": action_penalty,
        "goal_bonus": goal_bonus,
        "fall_penalty": fall_pen,
    }
```

Notes on how this differs from the file it was removed from:

- `self.goal_dist` and `self.rot_dist` are computed in `_get_dones`, not here — the
  migration hoisted them out so the reward carries no load-bearing side effects.
- `self.goal_resets` is prepared by `_update_success_metrics`, also called from
  `_get_dones`.
- The removed version applied `self.cfg.action_penalty_scale` at the summation site
  (`total_reward = dist_rew + rot_rew + action_penalty * self.cfg.action_penalty_scale`)
  and folded the goal bonus / fall penalty in via `torch.where` on `total_reward`.
  The arithmetic above is identical; it is only regrouped so each component is a
  separate tensor that can be logged.

---

## Shadow Hand repose (vision) — `Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0`

`source/ard_tasks/ard_tasks/tasks/direct/shadow_hand_vision/shadow_hand_vision_env.py`

The vision task's reward is **byte-for-byte the same** as the state task's above.
Only the observations differ (a wrist camera feeding a CNN feature extractor
instead of privileged object state). Its reward scales come from
`ShadowHandVisionEnvCfg`, which overrides four of the state task's values:

| cfg field | state | vision |
|---|---|---|
| `dist_reward_scale` | `-10.0` | `-10.0` |
| `rot_reward_scale` | `1.0` | `1.0` |
| `rot_eps` | `0.1` | `0.1` |
| `action_penalty_scale` | `-0.0002` | `-0.0002` |
| `reach_goal_bonus` | `250` | `250` |
| `fall_penalty` | `0` | **`-50`** |
| `fall_dist` | `0.24` | `0.24` |
| `success_tolerance` | `0.1` | **`0.4`** |
| `max_consecutive_success` | `0` | **`50`** |

Use the state task's code block verbatim.

---

## The ARD reward contract

Every task env now splits the reward in two:

```python
def _get_rewards(self) -> torch.Tensor:
    """Framework hook. NOT an ARD edit target."""
    total_reward, reward_components = self.compute_reward()
    log_reward_components(self, total_reward, reward_components)
    return total_reward

def compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """<<< ARD EDIT TARGET >>> — the reward workspace."""
    ...
    return total_reward, {"name": component_tensor, ...}
```

- `compute_reward` is the **only** method ARD rewrites. It returns two things,
  exactly as in Eureka: the per-env total reward of shape `(num_envs,)`, and a dict
  naming each individual reward component (also `(num_envs,)` each).
- `log_reward_components` (`ard_tasks.utils.reward_logging`) reduces each component
  to its mean over envs and writes it to `self.extras["log"]` as `components_<name>`, plus
  the aggregate as `components_total`. IsaacLab's rl_games wrapper renames `log` ->
  `episode`, and rl_games' `IsaacAlgoObserver` writes each key to TensorBoard as
  `Episode/components_<name>`. That is how the LLM gets to see, iteration after iteration,
  what each component it wrote is actually doing during training.
- The fixed evaluation metric `fitness_function` is logged from `_get_dones`,
  outside both methods, so an ARD edit can never alter the score it is judged on.
