# IsaacLab image for running ARD tasks on CARES GPU machines (and the PCS runner).
#
# The stock nvcr.io/nvidia/isaac-lab image only runs as root/isaac-sim: its
# top-level /isaac-sim directory is mode 0750, so a container started as a
# non-root host user (the CARES rule `-u $(id -u):$(id -g)`) can't even traverse
# it to reach Python. This image, built once as root, fixes that:
#   1. pre-installs ard_tasks against IsaacLab's bundled interpreter, and
#   2. opens just the /isaac-sim gate (everything inside is already world-
#      readable, so no costly recursive chmod of the ~17GB tree is needed).
# Runtime writes (caches, logs) go to HOME, which the runner points at the
# writable student_data mount.
#
# Build (from the repo root):
#   docker pull nvcr.io/nvidia/isaac-lab:2.3.2
#   docker build -t pcs-isaaclab-ard:2.3.2 .
FROM nvcr.io/nvidia/isaac-lab:2.3.2

USER root

# Pre-install the ARD tasks (only extra dependency is psutil, already present),
# so the job never needs to pip-install at runtime — which would fail under the
# non-root `-u` rule anyway.
COPY source /opt/ard-isaaclab-tasks/source
COPY scripts /opt/ard-isaaclab-tasks/scripts
RUN /isaac-sim/python.sh -m pip install -e /opt/ard-isaaclab-tasks/source/ard_tasks \
 && chmod o+rx /isaac-sim \
 && find /isaac-sim -xdev -type f ! -perm -o+r -exec chmod o+r {} + \
 && chmod -R o+rX /opt/ard-isaaclab-tasks
