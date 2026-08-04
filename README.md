# ard-isaaclab-tasks

**Contents:** [Introduction](#introduction) · [Compatibility](#compatibility) · [Installation](#installation) · [How to use](#how-to-use) · [Repository layout](#repository-layout)

## Introduction

### What is it?

`ard-isaaclab-tasks` is the IsaacLab task substrate for the **Autonomous RL Designer (ARD)**: a research framework that uses an LLM to generate reward functions, trains them with PPO in IsaacLab, and reflects on VLM evaluations of rollout videos to iterate. This repo only holds the RL training side, the IsaacLab tasks ARD trains against. The LLM/VLM loop itself lives elsewhere.

This is an **external IsaacLab project** (generated via `isaaclab.sh --new`): the source tree lives outside the core IsaacLab repository and is installed as an editable extension against an existing IsaacLab 2.3.2 install.

### Current task support

Every task here is copied from the official IsaacLab 2.3.2 source and registered under an `Isaac-ARD-*` ID. In each task, all reward logic lives in a single `_get_rewards` method, the one method ARD is allowed to rewrite. See [Preparing a workspace for ARD](#preparing-a-workspace-for-ard) for how that edit contract works.

| Task ID | Description |
| --- | --- |
| `Isaac-ARD-Cartpole-v0` | Classic cartpole balancing, single-agent and low-DoF. The fast smoke-test task. |
| `Isaac-ARD-Repose-Cube-Shadow-Direct-v0` | Shadow Hand reposes a cube to a target orientation from **state** observations. Config, spaces and agent hyperparameters are a verbatim migration of the official `Isaac-Repose-Cube-Shadow-Direct-v0` benchmark. |
| `Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0` | Shadow Hand reposes a cube to a target orientation from **vision** (TiledCamera + CNN feature extractor). Verbatim migration of the official `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` benchmark. |

A play/eval variant `Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-Play-v0` (fewer envs, CNN feature extractor in inference mode) is also registered for checking a trained policy rather than training one. An earlier, larger multi-task suite (Humanoid, Franka-Cabinet, Allegro-Repose, Forge-NutThread, Shadow-Hand-Over) was removed in v0.3.0 and remains available at tag `v0.2.0`.

> The Shadow tasks register under the `Isaac-ARD-` prefix (not the official `Isaac-Repose-...` IDs) solely to avoid a gym duplicate-registration clash. `scripts/train.py` imports `isaaclab_tasks`, which already registers the official IDs. Every cfg entry point and agent hyperparameter is otherwise identical to the official benchmark; only reward composition differs.

List the installed `Isaac-ARD-*` IDs at runtime:

```bash
python scripts/list_envs.py
# or: <isaaclab>/isaaclab.sh -p scripts/list_envs.py
```

`scripts/list_envs.py` filters the registry by the `"Isaac-ARD-"` prefix, so it prints exactly the tasks this repo registers.

## Compatibility

- IsaacLab **2.3.2** (Direct workflow only). The repo is pinned to this version because PPO hyperparameters and observation/action spaces are copied verbatim from NVIDIA's official 2.3.2 benchmarks; deviations would invalidate the reward-design alignment that motivates ARD.
- RL library: `rl_games`.
- License: MIT (see [`LICENSE`](LICENSE)).

## Installation

1. Install Isaac Lab 2.3.2 by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/isaaclab_pip_installation.html). The conda or uv install is recommended because it puts `python` on PATH with Isaac Sim's bundled interpreter; substitute `<isaaclab>` below with the path to your IsaacLab clone (e.g. `~/IsaacLab`).

2. Clone this repo **outside** the IsaacLab directory.

3. Install this project as an editable extension using a Python interpreter that has Isaac Lab installed:

   ```bash
   conda activate env_isaaclab
   cd ard-isaaclab-tasks
   python -m pip install -e source/ard_tasks
   ```

## How to use

### Directly running

There are three ways to run a task, in increasing order of setup: locally with a bare `python`, locally in Docker, and on the shared CARES HPC cluster.

#### Local running

Once the conda/uv env is activated, `python` already points at Isaac Sim's interpreter:

```bash
python scripts/train.py --task Isaac-ARD-Cartpole-v0 --headless

# Shadow-Hand vision benchmark: a camera task, so --enable_cameras is required:
python scripts/train.py --task Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0 --headless --enable_cameras
```
Once training is done, evaluation can be observed using the `--video` flag. This spawns multiple cameras surrounding the task and saves the video to log
```bash
# Record video during evaluation
python scripts/play.py --task Isaac-ARD-Repose-Cube-Shadow-Direct-v0 --headless --video --num_envs 1
```

Standard flags pass through to the rl_games runner: `--num_envs`, `--seed`, `--headless`, `--video`, `--checkpoint`, `--max_iterations`.

**Early stopping.** `--early-stop-patience N` stops a run once the `fitness_function` metric (see [Preparing a workspace for ARD](#preparing-a-workspace-for-ard)) hasn't improved for `N` training epochs -- useful in automated pipelines that evaluate many reward-function candidates, where a plateaued run is wasted compute. It defaults to `-1`, which disables early stopping so a run always goes to full duration:

```bash
python scripts/train.py --task Isaac-ARD-Cartpole-v0 --headless --early-stop-patience 20
```

When it triggers, the run logs the reason, the best `fitness_function` value and the epoch it occurred at, then stops exactly as it would on hitting `max_iterations` (same checkpointing).

**Dummy-agent sanity checks:** the scaffolded `scripts/zero_agent.py` and `scripts/random_agent.py` (from the IsaacLab `--new` template) run each env with a zero or random policy. Handy for confirming an env constructs and steps without rl_games in the loop:

```bash
python scripts/zero_agent.py --task Isaac-ARD-Cartpole-v0 --num_envs 4
python scripts/random_agent.py --task Isaac-ARD-Cartpole-v0 --num_envs 4
```

#### Local docker running

The CARES GPU machines run every job in a container as your own (non-root) user. The stock `nvcr.io/nvidia/isaac-lab` image only runs as root, so this repo ships a small derived image that pre-installs `ard_tasks` and opens IsaacLab to a non-root uid. See [`Dockerfile`](Dockerfile) for the details.

**Build:**

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2     # base (no NGC login needed)
docker build -t pcs-isaaclab-ard:2.3.2 .       # from the repo root
```

The build adds only a few MB on top of the base, so it finishes in seconds once the base image is present.

**Run directly (sanity check), as your own user, GPU attached:**

```bash
docker run --rm --gpus all \
  --entrypoint sh pcs-isaaclab-ard:2.3.2 \
  -c '/workspace/isaaclab/isaaclab.sh -p /opt/ard-isaaclab-tasks/scripts/train.py \
        --task Isaac-ARD-Cartpole-v0 --headless'
```

#### HPC running

The [CARES HPC Scheduler](https://uoa-cares.github.io/hpc-client/) runs jobs on a shared GPU cluster by **pulling a prebuilt image** from the CARES registry, rather than building one per job like PCS does. That one difference drives most of the workflow: you push an image ahead of time, artifacts are only preserved from `/workspace/output`, and a reward edit needs a re-push (`scripts/hpc_push.sh`, ~1 MB per iteration) before it reaches the cluster.

Once you have an account and the `hpc-client` CLI configured, submitting a seed sweep is one command:

```bash
python scripts/hpc_submit.py --task Isaac-ARD-Cartpole-v0 --seeds 0 1 2 \
    --max-iterations 150 --max-runtime-hours 2
```

**See [`docs/HPC.md`](docs/HPC.md) for the full walkthrough:** getting an account, installing/configuring `hpc-client`, trusting the CARES registry, building/pushing the image, submitting and monitoring jobs, and collecting results from the NAS.

A few gotchas worth knowing before you submit (full detail in [`docs/HPC.md`](docs/HPC.md#things-that-will-bite-you)):

- **`--num_envs` is not free to choose.** An arbitrary value breaks an rl_games assertion unless you also adjust the agent cfg.
- **The vision task does not run on the HPC yet** (an RTX renderer issue on the worker GPUs). Run it locally or in Docker instead.
- **`max_runtime_hours` is a hard kill**, so overestimate it.

### Preparing a workspace for ARD

**The `_get_rewards` contract.** Every task's environment class exposes its reward computation in a single method with a fixed signature, so ARD's AST-level code generator can rewrite it unambiguously:

```python
# Cartpole (single-agent)
def _get_rewards(self) -> torch.Tensor:
    """Compute per-env scalar reward.

    All reward shaping, dense/sparse signals, and termination bonuses
    must be computed inside this method. Return shape: (num_envs,).
    This method is the sole edit target for the ARD framework.
    """
    ...
```

No reward logic lives outside `_get_rewards` in any task, including the Shadow Hand benchmarks. Hyperparameters, observation spaces, action spaces, and termination conditions stay unchanged from the official IsaacLab 2.3.2 source. `_get_rewards` is the *only* thing ARD edits. Success tracking and the `fitness_function` evaluation metric live outside `_get_rewards`, so an ARD edit can never touch the score it's judged on.

**`ard_meta.yaml`.** Each task directory carries an `ard_meta.yaml`: the task id, the path to its env file, and a natural-language task description, meant to be handed to the external ARD framework directly as its `--taskconfig`:

```yaml
# source/ard_tasks/ard_tasks/tasks/direct/cartpole/ard_meta.yaml
task: "Isaac-ARD-Cartpole-v0"
env_file: "source/ard_tasks/ard_tasks/tasks/direct/cartpole/cartpole_env.py"
description: >
  Balance a pole upright on a cart by applying horizontal forces to the cart. ...
```

**Starting from a clean slate.** ARD is meant to regenerate each task's reward from scratch, not edit an existing one. So before a run, purge the reference implementations onto a dedicated `workspace` branch:

```bash
scripts/ard_workspace.sh                  # branch off the current HEAD
scripts/ard_workspace.sh --base main       # branch off a specific ref
scripts/ard_workspace.sh --force           # recreate an existing `workspace` branch
scripts/ard_workspace.sh --dry-run         # preview the git/purge commands only
```

This creates (or recreates) a `workspace` branch and blanks every `_get_rewards` body on it down to `return torch.zeros(self.num_envs, device=self.device)`, so no reference reward leaks into the LLM's prompt. It refuses to run on a dirty working tree.

## Repository layout

The layout mirrors what `<isaaclab>/isaaclab.sh --new` produces for an external project:

```
ard-isaaclab-tasks/
├── source/ard_tasks/              # editable extension (`pip install -e`)
│   └── ard_tasks/tasks/direct/
│       ├── cartpole/               # Isaac-ARD-Cartpole-v0 + ard_meta.yaml
│       ├── shadow_hand/            # Isaac-ARD-Repose-Cube-Shadow-Direct-v0 (state) + ard_meta.yaml
│       └── shadow_hand_vision/     # Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0 (+ -Play-v0) + ard_meta.yaml
├── scripts/
│   ├── train.py                    # rl_games train entry point
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── run_all_experiments.sh      # run every Isaac-ARD-* task sequentially, log each
│   ├── ard_workspace.sh            # create/refresh the `workspace` branch (rewards purged)
│   ├── purge_rewards.py            # AST tool that blanks `_get_rewards` bodies
│   ├── pcs_entrypoint.sh           # image CMD under PCS (artifacts -> /work/logs)
│   ├── hpc_entrypoint.sh           # job command on CARES HPC (artifacts -> /workspace/output)
│   ├── hpc_push.sh                 # build + push the image to the CARES registry
│   └── hpc_submit.py               # submit seed sweeps to the HPC scheduler
├── hpc/                             # ready-to-edit job files for `hpc-client submit`
│   ├── cartpole.json
│   └── shadow-vision.json
├── docs/HPC.md                      # full CARES HPC Scheduler walkthrough
├── quickstart.sh                    # PCS job command: installs ard_tasks, runs train.py
├── Dockerfile                       # CARES non-root IsaacLab image (see "Local docker running")
└── source/ard_tasks/{setup.py,pyproject.toml,config/extension.toml,...}
```
