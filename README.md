# ard-isaaclab-tasks

`ard-isaaclab-tasks` is the IsaacLab task substrate for the **Autonomous RL Designer (ARD)**: a research framework that uses an LLM to generate reward functions, trains them with PPO in IsaacLab, and reflects on VLM evaluations of rollout videos to iterate. This repo holds only the RL training side — six tasks copied verbatim from the official IsaacLab 2.3.X source, registered under `Isaac-ARD-*` IDs, with each task's reward computation isolated in a single `_get_rewards` method that ARD's code generator targets via AST rewriting.

This is an **external Isaac Lab project** (generated via `isaaclab.sh --new`): the source tree lives outside the core IsaacLab repository and is installed as an editable extension against an existing IsaacLab 2.3.X install.

## Compatibility

- IsaacLab **2.3.X** (Direct workflow only). The repo is pinned to this version because PPO hyperparameters and observation/action spaces are copied verbatim from NVIDIA's official 2.3.X benchmarks; deviations would invalidate the reward-design alignment that motivates ARD.
- RL library: `rl_games`.
- License: MIT (see [`LICENSE`](LICENSE)).

## Installation

1. Install Isaac Lab 2.3.X by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). The conda or uv install is recommended because it puts `python` on PATH with Isaac Sim's bundled interpreter; substitute `<isaaclab>` below with the path to your IsaacLab clone (e.g. `~/IsaacLab`).

2. Clone this repo **outside** the IsaacLab directory.

3. Install this project as an editable extension using a Python interpreter that has Isaac Lab installed:

   ```bash
   # If using the recommended conda env (e.g. `conda activate env_isaaclab`)
   cd ard-isaaclab-tasks
   python -m pip install -e source/ard_tasks

   # Otherwise use IsaacLab's Python wrapper directly:
   <isaaclab>/isaaclab.sh -p -m pip install -e source/ard_tasks
   ```

## Registered tasks

| Task ID | Description |
| --- | --- |
| `Isaac-ARD-Cartpole-v0` | Classic cartpole balancing — single-agent, low-DoF baseline. |
| `Isaac-ARD-Humanoid-v0` | 21-DoF humanoid forward locomotion on flat terrain. |
| `Isaac-ARD-Franka-Cabinet-v0` | Franka arm opens a cabinet drawer (contact-rich manipulation). |
| `Isaac-ARD-Allegro-Repose-v0` | Allegro hand reposes a cube to a target orientation (dexterous, high-DoF). |
| `Isaac-ARD-Forge-NutThread-v0` | Forge nut-threading on a bolt — contact-rich, sparse, long-horizon. |
| `Isaac-ARD-Shadow-Hand-Over-v0` | Two Shadow hands hand off a cube (multi-agent, `DirectMARLEnv`). |

The six tasks were chosen to span a 7-dimension binary feature space (high-DoF, contact-rich, sparse reward, long-horizon, manipulation, dexterous hand, multi-agent) with no two tasks sharing a feature signature.

List the installed `Isaac-ARD-*` IDs at runtime:

```bash
python scripts/list_envs.py
# or: <isaaclab>/isaaclab.sh -p scripts/list_envs.py
```

> The bundled `scripts/list_envs.py` filters by the `"Template-"` prefix. Either edit that prefix to `"Isaac-ARD-"`, or simply: `python -c "import gymnasium as gym; import ard_tasks; print([k for k in gym.envs.registry if 'ARD' in k])"`.

## Training

Once the conda/uv env is activated, `python` already points at Isaac Sim's interpreter:

```bash
python scripts/train.py --task Isaac-ARD-Cartpole-v0 --headless
python scripts/train.py --task Isaac-ARD-Humanoid-v0 --headless
python scripts/train.py --task Isaac-ARD-Franka-Cabinet-v0 --headless
python scripts/train.py --task Isaac-ARD-Allegro-Repose-v0 --headless
python scripts/train.py --task Isaac-ARD-Forge-NutThread-v0 --headless
python scripts/train.py --task Isaac-ARD-Shadow-Hand-Over-v0 --headless
```

If Isaac Lab is not on PATH, use the wrapper instead — e.g.:

```bash
<isaaclab>/isaaclab.sh -p scripts/train.py --task Isaac-ARD-Cartpole-v0 --headless
```

Standard flags pass through to the rl_games runner: `--num_envs`, `--seed`, `--headless`, `--video`, `--checkpoint`, `--max_iterations`.

### Dummy-agent sanity checks

The scaffolded `scripts/zero_agent.py` and `scripts/random_agent.py` (from the IsaacLab `--new` template) run each env with a zero or random policy — useful for confirming an env constructs and steps without rl_games in the loop:

```bash
python scripts/zero_agent.py --task Isaac-ARD-Cartpole-v0 --num_envs 4
python scripts/random_agent.py --task Isaac-ARD-Cartpole-v0 --num_envs 4
```

## Reward isolation (ARD contract)

Every task's environment class exposes its reward computation in a single method with a fixed signature so ARD's AST-level code generator can rewrite it unambiguously:

```python
# Single-agent tasks (Cartpole, Humanoid, Franka-Cabinet, Allegro-Repose, Forge-NutThread)
def _get_rewards(self) -> torch.Tensor:
    """Compute per-env scalar reward.

    All reward shaping, dense/sparse signals, and termination bonuses
    must be computed inside this method. Return shape: (num_envs,).
    This method is the sole edit target for the ARD framework.
    """
    ...

# Multi-agent task (Shadow-Hand-Over)
def _get_rewards(self) -> dict[str, torch.Tensor]:
    """Compute per-agent per-env scalar rewards.

    Keys are agent IDs (matching the env's agent set); each value
    is a tensor of shape (num_envs,). All reward shaping must live here.
    This method is the sole edit target for the ARD framework.
    """
    ...
```

No reward logic lives outside `_get_rewards`. Hyperparameters, observation spaces, action spaces, and termination conditions are unchanged from the official IsaacLab 2.3.X source — only `_get_rewards` is an ARD edit target.

## Repository layout

The layout mirrors what `<isaaclab>/isaaclab.sh --new` produces for an external project:

```
ard-isaaclab-tasks/
├── source/ard_tasks/         # editable extension (`pip install -e`)
│   └── ard_tasks/tasks/direct/
│       ├── cartpole/         # verbatim copy + Isaac-ARD-* register call
│       ├── humanoid/
│       ├── franka_cabinet/
│       ├── allegro_hand/
│       ├── forge/            # depends on factory/
│       ├── shadow_hand_over/
│       ├── factory/          # parent env for Forge tasks (not registered)
│       ├── inhand_manipulation/  # parent env for Allegro (not registered)
│       └── locomotion/       # parent env for Humanoid (not registered)
├── scripts/
│   ├── train.py              # rl_games train entry point
│   ├── list_envs.py
│   ├── zero_agent.py
│   └── random_agent.py
└── source/ard_tasks/{setup.py,pyproject.toml,config/extension.toml,...}
```
