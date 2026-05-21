#!/bin/bash
# Train an ARD task inside the prepared Docker image (pcs-isaaclab-ard:2.3.2),
# run from the extracted codebase root.
#
# ard_tasks is already installed in the image, so there is no `pip install` here
# (a runtime install would fail under the CARES non-root `-u` rule). `python` is
# only a bash alias for root in the base image, so we call IsaacLab's interpreter
# wrapper directly.
#
# Usage: bash quickstart.sh [TASK_ID]      (default: Isaac-ARD-Cartpole-v0)
#        MAX_ITERATIONS=1500 bash quickstart.sh Isaac-ARD-Humanoid-v0
set -e
TASK="${1:-Isaac-ARD-Cartpole-v0}"
/workspace/isaaclab/isaaclab.sh -p scripts/train.py \
    --task "$TASK" --headless --max_iterations "${MAX_ITERATIONS:-100}"
echo "=== $TASK training complete ==="
