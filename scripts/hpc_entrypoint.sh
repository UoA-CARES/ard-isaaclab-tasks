#!/usr/bin/env bash
# Entrypoint for ARD IsaacLab training jobs on the CARES HPC Scheduler
# (https://uoa-cares.github.io/hpc-client/).
#
# The HPC scheduler differs from PCS (scripts/pcs_entrypoint.sh) in two ways that
# this script exists to absorb:
#
#   1. Outputs. The scheduler only preserves what a job writes to
#      /workspace/output; everything else in the container is discarded when the
#      job exits. train.py writes its rl_games run dir to `logs/rl_games/...`
#      *relative to the working directory*, so we cd into /workspace/output first
#      and the run lands at /workspace/output/logs/rl_games/... The scheduler then
#      copies it to the NAS at /cares-nas/hpc/outputs/<upi>/<job_id>.
#
#   2. Nothing under /workspace can be trusted. The scheduler mounts its own
#      things into that tree (/workspace/output, /workspace/datasets/<name>), and
#      the docs never say whether it binds those subpaths or mounts a volume over
#      /workspace itself -- which would hide the base image's IsaacLab at
#      /workspace/isaaclab. So this job depends on nothing under /workspace:
#      the Dockerfile relocates IsaacLab's importable source to /opt/isaaclab,
#      and we call Isaac Sim's interpreter at /isaac-sim/python.sh (the same
#      script isaaclab.sh -p ends up exec'ing). Verified locally under both mount
#      layouts. Everything below /workspace is treated purely as an output sink.
#
# Training flags arrive as this script's command-line arguments (the job
# `command`) and are forwarded verbatim to train.py via "$@". This differs from
# PCS: the CARES HPC Scheduler does NOT inject the job `env` block into the
# container, so configuration cannot travel as env vars -- hpc_submit.py packs it
# into the `command` instead. (Passing task/seed/etc via `env` here silently
# dropped them, and every job fell back to the default Isaac-ARD-Cartpole-v0.)
#
# As a fallback for a manual `docker run` with no args, the run can still be
# driven by environment variables:
#   TASK            task id                (default Isaac-ARD-Cartpole-v0)
#   MAX_ITERATIONS  PPO iterations         (--max_iterations, if set)
#   NUM_ENVS        parallel environments  (--num_envs, if set)
#   SEED            rng seed               (--seed, if set)
#   EXTRA_ARGS      raw args appended verbatim to train.py
#                   (the vision task needs EXTRA_ARGS="--enable_cameras")
# These always apply, read from the environment regardless of args:
#   WANDB_API_KEY   if set, enables --track and W&B logging
#   WANDB_PROJECT / WANDB_ENTITY / WANDB_NAME   W&B run metadata
#   OUTPUT_DIR      override the preserved output dir (default /workspace/output)
#   ARD_SRC         if set, prepended to PYTHONPATH -- see "Reward iteration" below
#
# Reward iteration (ARD): the scheduler *pulls a prebuilt image*, so unlike PCS it
# does not rebuild from your working tree. A reward edited in `_get_rewards` only
# reaches the cluster once the image is rebuilt and pushed (scripts/hpc_push.sh --
# the rebuild is incremental and pushes only a few MB). If you need to iterate
# without a push, put the edited `ard_tasks` package tree on the NAS as a dataset,
# request it in `required_datasets`, and set ARD_SRC to its mount path; it then
# takes import precedence over the copy baked into the image.
set -euo pipefail

PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
TRAIN="${TRAIN_SCRIPT:-/opt/ard-isaaclab-tasks/scripts/train.py}"
PLAY="${PLAY_SCRIPT:-/opt/ard-isaaclab-tasks/scripts/play.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"

if [ ! -x "$PYTHON" ]; then
  echo "[hpc] FATAL: Isaac Sim interpreter not found at $PYTHON" >&2
  exit 1
fi

# The scheduler normally creates this; mkdir keeps a plain `docker run` working.
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

if [ -n "${ARD_SRC:-}" ]; then
  export PYTHONPATH="${ARD_SRC}${PYTHONPATH:+:$PYTHONPATH}"
  echo "[hpc] ard_tasks override on PYTHONPATH: $ARD_SRC"
fi

args=(--headless)

# WANDB secrets, if present, come from the environment -- never the command line,
# where they would be visible in `hpc-client jobs`.
if [ -n "${WANDB_API_KEY:-}" ]; then
  args+=(--track)
  [ -n "${WANDB_PROJECT:-}" ] && args+=(--wandb-project-name "$WANDB_PROJECT")
  [ -n "${WANDB_ENTITY:-}" ]  && args+=(--wandb-entity "$WANDB_ENTITY")
  [ -n "${WANDB_NAME:-}" ]    && args+=(--wandb-name "$WANDB_NAME")
fi

# Determine execution mode from the first argument (default to train)
TARGET_SCRIPT="$TRAIN"
mode_label="train"

if [ "$#" -gt 0 ]; then
  case "$1" in
    play|eval)
      TARGET_SCRIPT="$PLAY"
      mode_label="play"
      if [ "$mode_label" = "play" ]; then
        baked_ckpt="/opt/ard-isaaclab-tasks/scripts/warm_start/checkpoint.pth"
        staged_dir="$OUTPUT_DIR/logs/rl_games/play/nn"
        mkdir -p "$staged_dir"
        cp "$baked_ckpt" "$staged_dir/model.pth"
        args+=(--checkpoint "$staged_dir/model.pth")
      fi
      shift
      ;;
    train)
      TARGET_SCRIPT="$TRAIN"
      mode_label="train"
      shift
      ;;
  esac
fi

if [ "$#" -gt 0 ]; then
  # Scheduler path: the training flags are the job `command` arguments.
  args+=("$@")
  task_label="(from command args)"
else
  # Fallback path (manual `docker run` with no args): derive flags from env vars.
  task_label="${TASK:-Isaac-ARD-Cartpole-v0}"
  args+=(--task "$task_label")
  [ -n "${NUM_ENVS:-}" ]       && args+=(--num_envs "$NUM_ENVS")
  [ -n "${SEED:-}" ]           && args+=(--seed "$SEED")
  if [ "$mode_label" = "train" ]; then
    [ -n "${MAX_ITERATIONS:-}" ] && args+=(--max_iterations "$MAX_ITERATIONS")
  else
    # Play-specific env var fallbacks
    [ -n "${CHECKPOINT:-}" ] && args+=(--checkpoint "$CHECKPOINT")
    [ "${ENABLE_VIDEO:-false}" = "true" ] || [ "${VIDEO:-false}" = "true" ] && args+=(--video)
  fi

  # EXTRA_ARGS is deliberately word-split so callers can pass multiple flags.
  # shellcheck disable=SC2206
  [ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})
fi

echo "[hpc] user=$(id -u):$(id -g)  cwd=$(pwd)"
echo "[hpc] mode=${mode_label}  script=${TARGET_SCRIPT}"
echo "[hpc] task=${task_label}  artifacts -> ${OUTPUT_DIR}/logs"
echo "[hpc] exec: ${PYTHON} ${TARGET_SCRIPT} ${args[*]}"
exec "$PYTHON" "$TARGET_SCRIPT" "${args[@]}"