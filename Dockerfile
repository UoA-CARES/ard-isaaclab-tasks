# IsaacLab image for running ARD tasks under the Parallel Coordination System
# (PCS, /home/lee/code/parallel_coordination_system) on CARES GPU machines.
#
# PCS is a deploy-by-Dockerfile job runner: a worker *builds this Dockerfile from
# the submitted project tarball for every job* (it never pulls a prebuilt image),
# then runs the image's CMD as the submitting (non-root) user with HOME and the
# working dir pointed at a per-job mount. The PCS contract this image satisfies:
#   - Dockerfile at the project root (this file).
#   - CMD does the work and exits (0 = success); see scripts/pcs_entrypoint.sh.
#   - All artifacts are written under the mount: train.py writes `logs/...`
#     relative to the working dir, so declare `output_paths: ["logs"]`.
#   - It runs non-root under `-u $(id -u):$(id -g)`; nothing here needs root or
#     writes outside the mount / $HOME at runtime.
#   - GPU device is chosen at runtime by IsaacLab; PCS controls `--gpus`.
#   - Secrets (e.g. WANDB_API_KEY) arrive via the job `env`, never baked in.
#
# Because the build runs per job from the submitted tarball, the editable install
# below binds `import ard_tasks` to THIS job's source tree — so a reward edited in
# `_get_rewards` takes effect on the next submission with no manual rebuild. (This
# resolves the "imports bind to the baked copy" caveat in the README.)
#
# The same image also serves the CARES HPC Scheduler, which does NOT build from a
# tarball — it pulls this image prebuilt from the registry, so there the code is
# whatever was baked in at push time (see scripts/hpc_entrypoint.sh and
# scripts/hpc_push.sh). HPC jobs override the CMD below with
# `bash /opt/ard-isaaclab-tasks/scripts/hpc_entrypoint.sh`, which writes artifacts
# to /workspace/output — the only directory the scheduler preserves.
#
# Two problems with the stock base image are fixed here, once, as root at build
# time (a runtime fix would fail under the non-root `-u` rule):
#   1. ard_tasks is pre-installed against IsaacLab's bundled interpreter, so the
#      job never pip-installs at runtime.
#   2. /isaac-sim is mode 0750, so a non-root uid can't even traverse it to reach
#      the interpreter at /isaac-sim/kit/python/...; we open just that gate
#      (its contents are already world-readable).
#
# Build (PCS does this automatically per job; manually it is):
#   docker pull nvcr.io/nvidia/isaac-lab:2.3.2
#   docker build -t pcs-isaaclab-ard:2.3.2 .
FROM nvcr.io/nvidia/isaac-lab:2.3.2

USER root

# Accept the Isaac Sim EULA and decline telemetry non-interactively, so the
# headless job never blocks on a prompt. (Building/submitting this image is the
# act of accepting the NVIDIA Isaac Sim license.)
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y

# Move IsaacLab's importable source out of /workspace.
#
# The base image editable-installs all six isaaclab* packages from
# /workspace/isaaclab/source/*, so `import isaaclab` reads straight out of
# /workspace. That is fine under PCS (which mounts /work), but the CARES HPC
# Scheduler mounts *its own* things into /workspace — /workspace/output and
# /workspace/datasets/<name>. If it mounts those as subpaths, /workspace/isaaclab
# survives; if it mounts a volume over /workspace itself, IsaacLab disappears and
# every job dies with `ModuleNotFoundError: No module named 'isaaclab'`. The HPC
# docs don't say which it is.
#
# Rather than bet on it, we take a copy of the 16 MB source tree at /opt/isaaclab
# and re-point the editable installs there, so imports never touch /workspace.
# The original /workspace/isaaclab is left in place, so isaaclab.sh and the PCS
# path keep working exactly as before. --no-deps keeps this offline: every
# dependency is already installed in the base image, we are only rewriting the
# .pth/finder files to the new location.
#
# This is its own layer, ahead of the ard_tasks COPY, so it is cached and never
# re-pushed when a reward function changes.
RUN cp -a /workspace/isaaclab /opt/isaaclab \
 && /isaac-sim/python.sh -m pip install --no-cache-dir --no-build-isolation --no-deps \
        -e /opt/isaaclab/source/isaaclab \
        -e /opt/isaaclab/source/isaaclab_assets \
        -e /opt/isaaclab/source/isaaclab_contrib \
        -e /opt/isaaclab/source/isaaclab_mimic \
        -e /opt/isaaclab/source/isaaclab_rl \
        -e /opt/isaaclab/source/isaaclab_tasks \
 && chmod -R o+rX /opt/isaaclab

# isaaclab.sh derives this itself, and no Python code reads it — but the base
# image exports it as /workspace/isaaclab, which may not exist at run time.
ENV ISAACLAB_PATH=/opt/isaaclab

# Pre-install the ARD tasks against IsaacLab's interpreter. toml/setuptools are
# already present in that interpreter, so --no-build-isolation keeps the build
# fast and offline. psutil (the only extra dep) is already present too.
COPY source /opt/ard-isaaclab-tasks/source
COPY scripts /opt/ard-isaaclab-tasks/scripts
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-build-isolation \
        -e /opt/ard-isaaclab-tasks/source/ard_tasks \
 # Open the /isaac-sim traversal gate for non-root uids (contents already o+r),
 # and ensure any newly written package files / our scripts are world-readable.
 && chmod o+rx /isaac-sim \
 && find /isaac-sim -xdev -type f ! -perm -o+r -exec chmod o+r {} + \
 && chmod -R o+rX /opt/ard-isaaclab-tasks \
 && chmod o+rx /opt/ard-isaaclab-tasks/scripts/*.sh \
 # Kit writes its data/cache/logs under /isaac-sim/kit/{data,logs,cache} (root-
 # owned; data/logs don't exist), which a non-root uid can't create. Pre-create
 # and open them so each ephemeral job writes Kit's transient state into its own
 # throwaway container layer — never the per-job mount, so clean-out stays cheap.
 && mkdir -p /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache \
 && chmod -R o+rwX /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache

# The base image's ENTRYPOINT is /isaac-sim/runheadless.sh, which launches a
# streaming Isaac Sim app and swallows any CMD as trailing args — so we reset it
# and run our training script directly (this is why the README's by-hand run
# forces `--entrypoint sh`; baking the reset in means PCS needs no such flag).
ENTRYPOINT []

# Env-driven training entrypoint. PCS sets `-u`, HOME and `-w` at run time, and
# may override this CMD via the job `command`. Defaults to Isaac-ARD-Cartpole-v0;
# select the task and tunables through the job `env` (TASK, MAX_ITERATIONS, ...).
CMD ["bash", "/opt/ard-isaaclab-tasks/scripts/pcs_entrypoint.sh"]
