#!/usr/bin/env bash
# Run every Isaac-ARD-* task with default training settings, sequentially,
# capturing stdout+stderr to a per-task log file under a timestamped folder.
#
# Usage:
#   scripts/run_all_experiments.sh                 # uses `python` on PATH
#   PYTHON=~/IsaacLab/isaaclab.sh scripts/run_all_experiments.sh
#       (when IsaacLab is not on PATH; the wrapper accepts `-p script.py ...`)
#
# Extra flags after the script name are forwarded to train.py, e.g.
#   scripts/run_all_experiments.sh --num_envs 1024
#
# -----------------------------------------------------------------------------
# Reference run (defaults, --headless), 2026-05-31 -> 2026-06-01
#   Task                           Result   Wall time
#   Isaac-ARD-Cartpole-v0          OK             92s  (~1.5 min)
#   Isaac-ARD-Humanoid-v0          OK            667s  (~11 min)
#   Isaac-ARD-Franka-Cabinet-v0    OK           1532s  (~26 min)
#   Isaac-ARD-Allegro-Repose-v0    OK           8467s  (~2.4 h)
#   Isaac-ARD-Forge-NutThread-v0   OK          30904s  (~8.6 h)
#   Isaac-ARD-Shadow-Hand-Over-v0  OK           3089s  (~51 min)
#   Total: 44751s (~12.4 h) -- all 6 tasks succeeded.
#
# Machine configuration for the run above:
#   OS:     Ubuntu 22.04.5 LTS (kernel 6.8.0-124-generic)
#   CPU:    Intel Core i9-10900 @ 2.80GHz (10 cores / 10 threads, up to 5.2 GHz)
#   RAM:    30 GiB
#   GPU:    NVIDIA GeForce RTX 3080, 10 GiB VRAM (driver 595.71.05)
#   CUDA:   11.8 (V11.8.89)
# -----------------------------------------------------------------------------

set -u  # do not set -e: one failing task should not abort the rest

TASKS=(
  Isaac-ARD-Cartpole-v0
  Isaac-ARD-Humanoid-v0
  Isaac-ARD-Franka-Cabinet-v0
  Isaac-ARD-Allegro-Repose-v0
  Isaac-ARD-Forge-NutThread-v0
  Isaac-ARD-Shadow-Hand-Over-v0
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS_DIR="$REPO_ROOT/experiment_runs/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Allow PYTHON override for IsaacLab's wrapper, which needs the `-p` flag.
PYTHON_BIN="${PYTHON:-python}"
if [[ "$PYTHON_BIN" == *isaaclab.sh ]]; then
  PY_CMD=("$PYTHON_BIN" -p)
else
  PY_CMD=("$PYTHON_BIN")
fi

SUMMARY="$RESULTS_DIR/summary.txt"
{
  echo "Run started: $(date -Iseconds)"
  echo "Results dir: $RESULTS_DIR"
  echo "Python:      ${PY_CMD[*]}"
  echo "Extra args:  $*"
  echo
} > "$SUMMARY"

overall_start=$SECONDS
failed=()

for task in "${TASKS[@]}"; do
  log_file="$RESULTS_DIR/${task}.log"
  echo "==> [$task] starting; log: $log_file"
  start=$SECONDS

  "${PY_CMD[@]}" scripts/train.py --task "$task" --headless "$@" \
    > "$log_file" 2>&1
  status=$?
  elapsed=$((SECONDS - start))

  if [[ $status -eq 0 ]]; then
    result="OK"
  else
    result="FAIL(exit=$status)"
    failed+=("$task")
  fi

  printf '%-32s %-15s %5ds  %s\n' "$task" "$result" "$elapsed" "$log_file" \
    | tee -a "$SUMMARY"
done

overall_elapsed=$((SECONDS - overall_start))
{
  echo
  echo "Run finished: $(date -Iseconds)"
  echo "Total time:   ${overall_elapsed}s"
  if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All ${#TASKS[@]} tasks succeeded."
  else
    echo "${#failed[@]}/${#TASKS[@]} tasks failed: ${failed[*]}"
  fi
} | tee -a "$SUMMARY"

[[ ${#failed[@]} -eq 0 ]]
