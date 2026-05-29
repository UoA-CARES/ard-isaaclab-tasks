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
 && chmod o+rx /opt/ard-isaaclab-tasks/scripts/pcs_entrypoint.sh \
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
