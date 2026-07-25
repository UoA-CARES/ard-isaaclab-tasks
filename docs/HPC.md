# CARES HPC Scheduler

Full walkthrough for running `ard-isaaclab-tasks` on the [CARES HPC Scheduler](https://uoa-cares.github.io/hpc-client/), a shared GPU cluster. This doc is linked from the main [README](../README.md#hpc-running); start there for the short version.

## How it differs from PCS

The HPC Scheduler's contract differs from PCS (the other CARES job runner, see the [Local docker running](../README.md#local-docker-running) section) in ways that drive everything below:

| | PCS | CARES HPC Scheduler |
| --- | --- | --- |
| Image | worker **builds** the Dockerfile from your submitted tarball, per job | worker **pulls a prebuilt image** from the CARES registry |
| Your code | whatever you submitted, reward edits are live | whatever was baked in at push time, **re-push after a reward edit** |
| Artifacts | anything under the per-job mount (`output_paths: ["logs"]`) | **only `/workspace/output`**; everything else is discarded |
| Job command | image `CMD` (`scripts/pcs_entrypoint.sh`) | `command` field → `scripts/hpc_entrypoint.sh` |

`scripts/hpc_entrypoint.sh` absorbs the HPC rules: it `cd`s into `/workspace/output` so `train.py` writes its rl_games run there, and the scheduler copies the result to the NAS at `/cares-nas/hpc/outputs/<upi>/<job_id>`.

## Step 1: Get an account

Accounts cannot be self-created. Ask the HPC admins on the [CARES Slack](https://caresuoa.slack.com/archives/C0B0AJKMCPM) with your name, UPI, university email, supervisor, and intended usage. You will get scheduler credentials and a CARES NAS account.

## Step 2: Install and configure the client

```bash
pip install -e /path/to/hpc-client            # provides `hpc-client` + the hpc_client package
hpc-client configure --scheduler-url http://<scheduler-host>:8080
hpc-client login <upi>                        # prompts for your password; stores a session in ~/.hpc
hpc-client whoami                             # verify: prints your username, role, email
```

Install it into the **same interpreter you run `scripts/hpc_submit.py` with** (the conda `env_isaaclab` env is fine, since `hpc_submit.py` never imports IsaacLab).

## Step 3: Trust the CARES registry (once per machine)

The registry is plain HTTP, so Docker refuses to push to it until it is declared insecure. Merge this into `/etc/docker/daemon.json`:

```json
{ "insecure-registries": ["130.216.238.2:5500"] }
```

```bash
sudo systemctl restart docker
curl -s http://130.216.238.2:5500/v2/_catalog     # should list the registry's images
```

`scripts/hpc_push.sh` checks this first and tells you what to do if it is missing.

## Step 4: Test the image locally, then push it

The cluster is shared and repeated failures earn a suspension, so run the image the way the scheduler will *before* pushing: same entrypoint, artifacts into a mounted `/workspace/output`:

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

The default tag is `130.216.238.2:5500/cli797_ard-isaaclab:2.3.2`; override it with `$IMAGE` here, and with `--image` (or `$ARD_HPC_IMAGE`) when submitting.

## Step 5: Submit

Each job runs the same image and differs only in its `command`, which carries the `train.py` flags to `hpc_entrypoint.sh` (forwarded verbatim via `"$@"`). Config travels in `command`, **not** the job `env` block: the scheduler does not inject a job's `env` into the container, so anything put there is silently dropped and the entrypoint falls back to its defaults (`Isaac-ARD-Cartpole-v0`, no seed). A seed sweep is therefore one command: no per-run job file, no rebuild.

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

Useful flags: `--num-envs`, `--extra-args "<raw train.py flags>"`, `--dataset <name>` (mounts a NAS dataset read-only at `/workspace/datasets/<name>`), `--name-prefix`, `--image`. Run `--help` for the full list.

Or hand-edit a job file and use the CLI directly. [`hpc/cartpole.json`](../hpc/cartpole.json) and [`hpc/shadow-vision.json`](../hpc/shadow-vision.json) are ready to copy:

```bash
hpc-client submit hpc/cartpole.json
```

The `command` is the whole control surface: `hpc_entrypoint.sh` followed by the `train.py` flags (the entrypoint adds `--headless`):

| Flag | Effect |
| --- | --- |
| `--task` | task id (default `Isaac-ARD-Cartpole-v0` if omitted) |
| `--max_iterations` | PPO iterations |
| `--num_envs` | parallel envs (see the caveat below) |
| `--seed` | rng seed |
| `--enable_cameras` | required by the vision task |

The entrypoint also reads a few **environment** variables: `WANDB_API_KEY` (+`WANDB_PROJECT`/`WANDB_ENTITY`/`WANDB_NAME`), `ARD_SRC` (prepend to `PYTHONPATH` so an edited `ard_tasks` overrides the baked copy), and `OUTPUT_DIR` (default `/workspace/output`). These configure the *container*, not the run. Since the scheduler does not forward the job `env` block, they are reliable only under a local `docker run -e ...`; on the scheduler, W&B tracking and the `ARD_SRC` override are not available this way.

## Step 6: Monitor

```bash
hpc-client jobs                # your jobs and their status
hpc-client logs <job_id>       # container stdout
hpc-client wait <job_id>       # block until it reaches a terminal state
hpc-client cancel <job_id>     # stop a job
hpc-client status              # cluster queue depth / idle workers
```

The web portal at `http://<scheduler-host>:8080` shows the same thing.

## Step 7: Collect results

Everything the job wrote to `/workspace/output` is copied to the NAS at `/cares-nas/hpc/outputs/<upi>/<job_id>`, preserving the directory structure, so a run lands as `logs/rl_games/<task>/<timestamp>/nn/*.pth`. Download it from `http://130.216.238.2:5000`, or mount the share:

```bash
mkdir -p ~/hpc_outputs
sudo mount -t cifs //130.216.238.2/outputs ~/hpc_outputs -o username=<upi>
```

## ARD reward iteration on the HPC

The scheduler pulls the image, so **an edited `_get_rewards` does not reach the cluster until you re-push.** Two ways to close the loop:

1. **Re-push (recommended).** `scripts/hpc_push.sh` after each edit. The IsaacLab layers are cached, so only the `ard_tasks` source and its editable-install layer change: about **1 MB per iteration**, not the 17 GB base.
2. **Ship the source as a dataset.** Upload the edited `ard_tasks` package tree to the NAS as a dataset, request it with `--dataset <name>`, and point `ARD_SRC` at its mount (`/workspace/datasets/<name>`). The entrypoint puts it at the front of `PYTHONPATH`, so it takes import precedence over the baked copy. No rebuild, no push. (The HPC docs prefer code to live in the image, so treat this as the fast-iteration escape hatch rather than the default.) **Caveat:** `ARD_SRC` is an env var, and the scheduler does not forward the job `env` block into the container, so `--env ARD_SRC=...` will not take effect there. This hatch works only under a local `docker run -e ARD_SRC=...`. Re-push is the reliable path on the cluster.

## Things that will bite you

- **`--num_envs` is not free to choose.** rl_games asserts `batch_size % minibatch_size == 0`, and the minibatch is tuned for the task's default env count. An arbitrary `NUM_ENVS` (e.g. `64` on Cartpole) dies at startup with a bare `AssertionError`. Leave it unset unless you also adjust the agent cfg.
- **The vision task needs `--enable_cameras`.** `hpc_submit.py` adds it automatically for `*-Vision-*` tasks; a hand-written job file must append `--enable_cameras` to its `command` itself.
- **The vision task does not run on the HPC yet.** It runs perfectly locally and under Docker, but on the CARES cluster the RTX renderer segfaults at Isaac Sim startup (a worker GPU/driver issue). We may add HPC support in the future, but it is not on our schedule right now. Run the vision benchmark locally for now.
- **`max_runtime_hours` is a hard kill**, so overestimate it. Cartpole at defaults is ~1.5 min; the Shadow benchmarks are hours.
- **Job limits.** 50 active jobs per user. A postgraduate account runs 5 in the Normal tier and 2 in Overflow; the rest are Opportunistic and can be **preempted** by higher-priority jobs.
- **Nothing importable lives under `/workspace`.** IsaacLab is relocated to `/opt/isaaclab` at build time, because the HPC docs never say whether the scheduler bind-mounts `/workspace/output` as a subpath or mounts a volume over `/workspace` itself. The latter would hide the base image's IsaacLab and fail every job with `ModuleNotFoundError: No module named 'isaaclab'`. Both layouts are verified to train. Don't reintroduce a runtime dependency on `/workspace/isaaclab` here (the PCS path still uses it, and that copy is still in the image).

> **Note:** the published HPC docs show a `submit_job` / `wait_for_job` / `get_logs` Python API, but the shipped `hpc_client` actually exposes `submit` / `wait` / `logs`. `scripts/hpc_submit.py` uses the real names.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: No module named 'isaaclab'` | The image predates the `/opt/isaaclab` relocation. Rebuild and re-push. |
| `AssertionError` in `rl_games/common/a2c_common.py` | `NUM_ENVS` breaks `batch_size % minibatch_size == 0`. |
| Job runs but the NAS output is empty | Something wrote outside `/workspace/output`. Only that directory is preserved. |
| Old rewards still training | The image was not re-pushed after the edit (see above). |
| `docker push` fails with a TLS/HTTPS error | The registry is not in `insecure-registries` (Step 3). |
| Job never starts | The cluster is busy, or your jobs are in the Opportunistic tier. Check `hpc-client status`. |
