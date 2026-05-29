#!/usr/bin/env bash
# PCS entrypoint for ARD IsaacLab training jobs.
#
# This is the image CMD when the project is run by the Parallel Coordination
# System (/home/lee/code/parallel_coordination_system). PCS runs every job:
#   - non-root, with `-u $(id -u):$(id -g)` (the CARES contract), and
#   - with HOME and the working directory set to the per-job mount (default
#     /work), into which all artifacts must be written.
#
# train.py writes its rl_games run dir under `logs/rl_games/...` *relative to the
# working directory*, so with PCS's `-w /work` everything lands in /work/logs.
# Declare `logs` in the job's `output_paths` to get it back as an artifact.
#
# The run is driven entirely by environment variables (PCS injects these via the
# job's `env` field), so the image CMD never needs editing between runs:
#   TASK            task id                (default Isaac-ARD-Cartpole-v0)
#   MAX_ITERATIONS  PPO iterations         (--max_iterations, if set)
#   NUM_ENVS        parallel environments  (--num_envs, if set)
#   SEED            rng seed               (--seed, if set)
#   EXTRA_ARGS      raw args appended verbatim to train.py
#   WANDB_API_KEY   if set, enables --track and W&B logging (PCS redacts it)
#   WANDB_PROJECT / WANDB_ENTITY / WANDB_NAME   W&B run metadata
#
# PCS's `command` field can still override this CMD entirely for one-off runs.
set -euo pipefail

ISAACLAB=/workspace/isaaclab/isaaclab.sh
TRAIN=/opt/ard-isaaclab-tasks/scripts/train.py

TASK="${TASK:-Isaac-ARD-Cartpole-v0}"

args=(--task "$TASK" --headless)
[ -n "${MAX_ITERATIONS:-}" ] && args+=(--max_iterations "$MAX_ITERATIONS")
[ -n "${NUM_ENVS:-}" ]       && args+=(--num_envs "$NUM_ENVS")
[ -n "${SEED:-}" ]           && args+=(--seed "$SEED")

if [ -n "${WANDB_API_KEY:-}" ]; then
  args+=(--track)
  [ -n "${WANDB_PROJECT:-}" ] && args+=(--wandb-project-name "$WANDB_PROJECT")
  [ -n "${WANDB_ENTITY:-}" ]  && args+=(--wandb-entity "$WANDB_ENTITY")
  [ -n "${WANDB_NAME:-}" ]    && args+=(--wandb-name "$WANDB_NAME")
fi

# EXTRA_ARGS is deliberately word-split so callers can pass multiple flags.
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})

echo "[pcs] user=$(id -u):$(id -g)  HOME=${HOME:-?}  cwd=$(pwd)"
echo "[pcs] task=${TASK}  artifacts -> $(pwd)/logs (declare 'logs' in output_paths)"
echo "[pcs] exec: isaaclab.sh -p train.py ${args[*]}"
exec "$ISAACLAB" -p "$TRAIN" "${args[@]}"
