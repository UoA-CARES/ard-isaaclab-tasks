# ard-isaaclab-tasks

`ard-isaaclab-tasks` is the IsaacLab task substrate for the **Autonomous RL Designer (ARD)**: a research framework that uses an LLM to generate reward functions, trains them with PPO in IsaacLab, and reflects on VLM evaluations of rollout videos to iterate. This repo holds only the RL training side. It registers tasks under `Isaac-ARD-*` IDs, all copied verbatim from the official IsaacLab 2.3.X source:

- **Cartpole** — a fast smoke test whose reward computation is isolated in a single `_get_rewards` method that ARD's code generator targets via AST rewriting.
- **Shadow-Hand cube-repose (state)** — a verbatim migration of the official `Isaac-Repose-Cube-Shadow-Direct-v0` benchmark. It is a frozen reference benchmark: its settings, configuration and reward are unchanged from the official source and are **not** ARD edit targets.
- **Shadow-Hand cube-repose (vision)** — a verbatim migration of the official `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` benchmark (TiledCamera + CNN feature extractor). Also a frozen reference benchmark and **not** an ARD edit target.

The earlier multi-task suite was removed in v0.3.0 and remains available at tag `v0.2.0`.

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
| `Isaac-ARD-Cartpole-v0` | Classic cartpole balancing — single-agent, low-DoF baseline. ARD reward edit target. |
| `Isaac-ARD-Repose-Cube-Shadow-Direct-v0` | Shadow Hand reposes a cube to a target orientation from **state** observations. Verbatim migration of the official `Isaac-Repose-Cube-Shadow-Direct-v0` benchmark — frozen, not an ARD edit target. |
| `Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0` | Shadow Hand reposes a cube to a target orientation from **vision** (TiledCamera + CNN feature extractor). Verbatim migration of the official `Isaac-Repose-Cube-Shadow-Vision-Direct-v0` benchmark — frozen, not an ARD edit target. |

A play/eval variant `Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-Play-v0` (fewer envs) is also registered. The earlier multi-task suite (Humanoid, Franka-Cabinet, Allegro-Repose, Forge-NutThread, Shadow-Hand-Over) was removed in v0.3.0 and remains available at tag `v0.2.0`.

> The Shadow tasks register under the `Isaac-ARD-` prefix (not the official `Isaac-Repose-...` IDs) solely to avoid a gym duplicate-registration clash — `scripts/train.py` imports `isaaclab_tasks`, which already registers the official IDs. Every setting, cfg entry point and agent hyperparameter is otherwise identical to the official benchmark.

List the installed `Isaac-ARD-*` IDs at runtime:

```bash
python scripts/list_envs.py
# or: <isaaclab>/isaaclab.sh -p scripts/list_envs.py
```

`scripts/list_envs.py` filters the registry by the `"Isaac-ARD-"` prefix, so it prints exactly the tasks this repo registers.

## Training

Once the conda/uv env is activated, `python` already points at Isaac Sim's interpreter:

```bash
python scripts/train.py --task Isaac-ARD-Cartpole-v0 --headless

# Shadow-Hand vision benchmark — a camera task, so --enable_cameras is required:
python scripts/train.py --task Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0 --headless --enable_cameras
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

## Docker image (CARES GPU machines / PCS)

The CARES GPU machines run every job in a container as your own (non-root) user.
The stock `nvcr.io/nvidia/isaac-lab` image only runs as root, so this repo ships
a small derived image that pre-installs `ard_tasks` and opens IsaacLab to a
non-root uid. See [`Dockerfile`](Dockerfile) for the details.

### Build

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.2     # base (no NGC login needed)
docker build -t pcs-isaaclab-ard:2.3.2 .       # from the repo root
```

The build adds only a few MB on top of the base, so it finishes in seconds once
the base image is present. The tag is arbitrary, but it must match the
`docker_image` you submit to the runner.

### Run

The image **must** be run with the CARES flags. Key points specific to this image:

- Mount your `student_data` at a path **other than `/workspace`** — the image
  installs IsaacLab at `/workspace/isaaclab`, so mounting over `/workspace` hides
  it. Use e.g. `/student_data`.
- Call IsaacLab's interpreter wrapper (`/workspace/isaaclab/isaaclab.sh -p`);
  bare `python`/`pip` are only root bash aliases in the base image.
- `train.py` accepts `--num_envs`, `--seed`, `--headless`, `--video`,
  `--checkpoint`, `--max_iterations` (note: there is **no** `--log_interval`).

```bash
# Direct run (sanity check), as your own user, GPU attached:
docker run --rm --gpus all \
  --entrypoint sh pcs-isaaclab-ard:2.3.2 \
  -c '/workspace/isaaclab/isaaclab.sh -p /opt/ard-isaaclab-tasks/scripts/train.py \
        --task Isaac-ARD-Cartpole-v0 --headless'
```

```
docker run --rm --gpus all \
  --entrypoint sh pcs-isaaclab-ard:2.3.2 \
  -c '/workspace/isaaclab/isaaclab.sh -p /opt/ard-isaaclab-tasks/scripts/train.py --headless'
```

When submitting through the PCS runner, package this repo as the codebase and set
the command to `bash quickstart.sh <TASK_ID>` (the runner extracts the codebase
into the working directory and applies the CARES flags for you).

> **Note (ARD reward iteration):** `ard_tasks` is installed into the image at
> build time, so `import ard_tasks` resolves to the *baked* copy, not the copy in
> a submitted codebase. If you iterate on reward functions, rebuild the image (or
> arrange for the edited package to take import precedence) so the new rewards are
> actually used.

## CARES HPC Scheduler

The [CARES HPC Scheduler](https://uoa-cares.github.io/hpc-client/) runs jobs on a
shared GPU cluster. Its contract differs from PCS in two ways that drive
everything below:

| | PCS | CARES HPC Scheduler |
| --- | --- | --- |
| Image | worker **builds** the Dockerfile from your submitted tarball, per job | worker **pulls a prebuilt image** from the CARES registry |
| Your code | whatever you submitted — reward edits are live | whatever was baked in at push time — **re-push after a reward edit** |
| Artifacts | anything under the per-job mount (`output_paths: ["logs"]`) | **only `/workspace/output`**; everything else is discarded |
| Job command | image `CMD` (`scripts/pcs_entrypoint.sh`) | `command` field → `scripts/hpc_entrypoint.sh` |

`scripts/hpc_entrypoint.sh` absorbs the HPC rules: it `cd`s into `/workspace/output`
so `train.py` writes its rl_games run there, and the scheduler copies the result to
the NAS at `/cares-nas/hpc/outputs/<upi>/<job_id>`.

### Step 1 — Get an account

Accounts cannot be self-created. Ask the HPC admins on the
[CARES Slack](https://caresuoa.slack.com/archives/C0B0AJKMCPM) with your name, UPI,
university email, supervisor, and intended usage. You will get scheduler credentials
and a CARES NAS account.

### Step 2 — Install and configure the client

```bash
pip install -e /path/to/hpc-client            # provides `hpc-client` + the hpc_client package
hpc-client configure --scheduler-url http://<scheduler-host>:8080
hpc-client login <upi>                        # prompts for your password; stores a session in ~/.hpc
hpc-client whoami                             # verify: prints your username, role, email
```

Install it into the **same interpreter you run `scripts/hpc_submit.py` with** (the
conda `env_isaaclab` env is fine — `hpc_submit.py` never imports IsaacLab).

### Step 3 — Trust the CARES registry (once per machine)

The registry is plain HTTP, so Docker refuses to push to it until it is declared
insecure. Merge this into `/etc/docker/daemon.json`:

```json
{ "insecure-registries": ["130.216.238.2:5500"] }
```

```bash
sudo systemctl restart docker
curl -s http://130.216.238.2:5500/v2/_catalog     # should list the registry's images
```

`scripts/hpc_push.sh` checks this first and tells you what to do if it is missing.

### Step 4 — Test the image locally, then push it

The cluster is shared and repeated failures earn a suspension, so run the image the
way the scheduler will *before* pushing — same entrypoint, artifacts into a mounted
`/workspace/output`:

```bash
mkdir -p /tmp/hpc-out
docker build -t 130.216.238.2:5500/<upi>_ard-isaaclab:2.3.2 .
docker run --rm --gpus all \
  -e TASK=Isaac-ARD-Cartpole-v0 -e MAX_ITERATIONS=3 \
  -v /tmp/hpc-out:/workspace/output \
  130.216.238.2:5500/<upi>_ard-isaaclab:2.3.2 \
  bash /opt/ard-isaaclab-tasks/scripts/hpc_entrypoint.sh
find /tmp/hpc-out -name '*.pth'      # a checkpoint here == the NAS will get one too
```

Then build and push in one step:

```bash
scripts/hpc_push.sh                                            # default tag (below)
IMAGE=130.216.238.2:5500/<upi>_ard-isaaclab:dev scripts/hpc_push.sh
scripts/hpc_push.sh --no-build                                 # push an already-built image
```

The default tag is `130.216.238.2:5500/cli797_ard-isaaclab:2.3.2`; override it with
`$IMAGE` here, and with `--image` (or `$ARD_HPC_IMAGE`) when submitting.

### Step 5 — Submit

Each job runs the same image and differs only in its `command`, which carries the
`train.py` flags to `hpc_entrypoint.sh` (forwarded verbatim via `"$@"`). Config
travels in `command`, **not** the job `env` block: the scheduler does not inject a
job's `env` into the container, so anything put there is silently dropped and the
entrypoint falls back to its defaults (`Isaac-ARD-Cartpole-v0`, no seed). A seed
sweep is therefore one command — no per-run job file, no rebuild:

```bash
# 3 seeds of Cartpole, 150 PPO iterations each, 2h runtime cap
python scripts/hpc_submit.py --task Isaac-ARD-Cartpole-v0 --seeds 0 1 2 \
    --max-iterations 150 --max-runtime-hours 2

# print the job specs without submitting anything
python scripts/hpc_submit.py --task Isaac-ARD-Cartpole-v0 --seeds 0 1 2 --dry-run

# submit and block until every job finishes, then dump their logs
python scripts/hpc_submit.py --task Isaac-ARD-Cartpole-v0 --seeds 0 1 2 --logs

# vision benchmark (--enable_cameras is added for you), 12h cap
python scripts/hpc_submit.py --task Isaac-ARD-Repose-Cube-Shadow-Vision-Direct-v0 \
    --seeds 1 --max-runtime-hours 12
```

Useful flags: `--num-envs`, `--extra-args "<raw train.py flags>"`, `--dataset <name>`
(mounts a NAS dataset read-only at `/workspace/datasets/<name>`), `--name-prefix`,
`--image`. Run `--help` for the full list.

Or hand-edit a job file and use the CLI directly — [`hpc/cartpole.json`](hpc/cartpole.json)
and [`hpc/shadow-vision.json`](hpc/shadow-vision.json) are ready to copy:

```bash
hpc-client submit hpc/cartpole.json
```

The `command` is the whole control surface — `hpc_entrypoint.sh` followed by the
`train.py` flags (the entrypoint adds `--headless`):

| Flag | Effect |
| --- | --- |
| `--task` | task id (default `Isaac-ARD-Cartpole-v0` if omitted) |
| `--max_iterations` | PPO iterations |
| `--num_envs` | parallel envs (see the caveat below) |
| `--seed` | rng seed |
| `--enable_cameras` | required by the vision task |

The entrypoint also reads a few **environment** variables — `WANDB_API_KEY`
(+`WANDB_PROJECT`/`WANDB_ENTITY`/`WANDB_NAME`), `ARD_SRC` (prepend to `PYTHONPATH`
so an edited `ard_tasks` overrides the baked copy), and `OUTPUT_DIR` (default
`/workspace/output`). These configure the *container*, not the run. Since the
scheduler does not forward the job `env` block, they are reliable only under a local
`docker run -e ...`; on the scheduler, W&B tracking and the `ARD_SRC` override are
not available this way.

### Step 6 — Monitor

```bash
hpc-client jobs                # your jobs and their status
hpc-client logs <job_id>       # container stdout
hpc-client wait <job_id>       # block until it reaches a terminal state
hpc-client cancel <job_id>     # stop a job
hpc-client status              # cluster queue depth / idle workers
```

The web portal at `http://<scheduler-host>:8080` shows the same thing.

### Step 7 — Collect results

Everything the job wrote to `/workspace/output` is copied to the NAS at
`/cares-nas/hpc/outputs/<upi>/<job_id>`, preserving the directory structure — so a
run lands as `logs/rl_games/<task>/<timestamp>/nn/*.pth`. Download it from
`http://130.216.238.2:5000`, or mount the share:

```bash
mkdir -p ~/hpc_outputs
sudo mount -t cifs //130.216.238.2/outputs ~/hpc_outputs -o username=<upi>
```

### ARD reward iteration on the HPC

The scheduler pulls the image, so **an edited `_get_rewards` does not reach the
cluster until you re-push.** Two ways to close the loop:

1. **Re-push (recommended).** `scripts/hpc_push.sh` after each edit. The IsaacLab
   layers are cached, so only the `ard_tasks` source and its editable-install layer
   change — about **1 MB per iteration**, not the 17 GB base.
2. **Ship the source as a dataset.** Upload the edited `ard_tasks` package tree to
   the NAS as a dataset, request it with `--dataset <name>`, and point `ARD_SRC` at
   its mount (`/workspace/datasets/<name>`). The entrypoint puts it at the front of
   `PYTHONPATH`, so it takes import precedence over the baked copy — no rebuild, no
   push. (The HPC docs prefer code to live in the image, so treat this as the
   fast-iteration escape hatch rather than the default.) **Caveat:** `ARD_SRC` is an
   env var, and the scheduler does not forward the job `env` block into the
   container, so `--env ARD_SRC=...` will not take effect there — this hatch works
   only under a local `docker run -e ARD_SRC=...`. Re-push is the reliable path on
   the cluster.

### Things that will bite you

- **`--num_envs` is not free to choose.** rl_games asserts
  `batch_size % minibatch_size == 0`, and the minibatch is tuned for the task's
  default env count. An arbitrary `NUM_ENVS` (e.g. `64` on Cartpole) dies at startup
  with a bare `AssertionError`. Leave it unset unless you also adjust the agent cfg.
- **The vision task needs `--enable_cameras`.** `hpc_submit.py` adds it automatically
  for `*-Vision-*` tasks; a hand-written job file must append `--enable_cameras` to
  its `command` itself.
- **The vision task does not run on the HPC yet.** It runs perfectly locally and
  under Docker, but on the CARES cluster the RTX renderer segfaults at Isaac Sim
  startup (a worker GPU/driver issue). We may add HPC support in the future, but it
  is not on our schedule right now — run the vision benchmark locally for now.
- **`max_runtime_hours` is a hard kill**, so overestimate it. Cartpole at defaults is
  ~1.5 min; the Shadow benchmarks are hours.
- **Job limits.** 50 active jobs per user. A postgraduate account runs 5 in the Normal
  tier and 2 in Overflow; the rest are Opportunistic and can be **preempted** by
  higher-priority jobs.
- **Nothing importable lives under `/workspace`.** IsaacLab is relocated to
  `/opt/isaaclab` at build time, because the HPC docs never say whether the scheduler
  bind-mounts `/workspace/output` as a subpath or mounts a volume over `/workspace`
  itself — the latter would hide the base image's IsaacLab and fail every job with
  `ModuleNotFoundError: No module named 'isaaclab'`. Both layouts are verified to
  train. Don't reintroduce a runtime dependency on `/workspace/isaaclab` here (the
  PCS path still uses it, and that copy is still in the image).

> **Note:** the published HPC docs show a `submit_job` / `wait_for_job` / `get_logs`
> Python API, but the shipped `hpc_client` actually exposes `submit` / `wait` /
> `logs`. `scripts/hpc_submit.py` uses the real names.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: No module named 'isaaclab'` | The image predates the `/opt/isaaclab` relocation — rebuild and re-push. |
| `AssertionError` in `rl_games/common/a2c_common.py` | `NUM_ENVS` breaks `batch_size % minibatch_size == 0`. |
| Job runs but the NAS output is empty | Something wrote outside `/workspace/output`. Only that directory is preserved. |
| Old rewards still training | The image was not re-pushed after the edit (see above). |
| `docker push` fails with a TLS/HTTPS error | The registry is not in `insecure-registries` (Step 3). |
| Job never starts | The cluster is busy, or your jobs are in the Opportunistic tier. Check `hpc-client status`. |

## Reward isolation (ARD contract)

Every task's environment class exposes its reward computation in a single method with a fixed signature so ARD's AST-level code generator can rewrite it unambiguously:

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

No reward logic lives outside `_get_rewards`. Hyperparameters, observation spaces, action spaces, and termination conditions are unchanged from the official IsaacLab 2.3.X source — only `_get_rewards` is an ARD edit target.

## Repository layout

The layout mirrors what `<isaaclab>/isaaclab.sh --new` produces for an external project:

```
ard-isaaclab-tasks/
├── source/ard_tasks/         # editable extension (`pip install -e`)
│   └── ard_tasks/tasks/direct/
│       ├── cartpole/         # verbatim copy + Isaac-ARD-* register call
│       ├── inhand_manipulation/  # shared in-hand env (InHandManipulationEnv)
│       └── shadow_hand/      # verbatim copy + Isaac-ARD-* register calls (state + vision)
├── scripts/
│   ├── train.py              # rl_games train entry point
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── pcs_entrypoint.sh     # image CMD under PCS (artifacts -> /work/logs)
│   ├── hpc_entrypoint.sh     # job command on CARES HPC (artifacts -> /workspace/output)
│   ├── hpc_push.sh           # build + push the image to the CARES registry
│   └── hpc_submit.py         # submit seed sweeps to the HPC scheduler
├── hpc/                      # ready-to-edit job files for `hpc-client submit`
│   ├── cartpole.json
│   └── shadow-vision.json
└── source/ard_tasks/{setup.py,pyproject.toml,config/extension.toml,...}
```
