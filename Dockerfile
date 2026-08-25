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

# Custom rl_games (UoA-CARES fork) in place of the base image's stock copy.
#
# The base image installs rl_games 1.6.1 as a normal site-packages distribution
# via isaaclab_rl's extras. Installing our fork under the same distribution name
# makes pip *uninstall* that copy first, so `import rl_games` resolves here and
# only here. Nothing in this build checks that, so if you are debugging a run
# that behaves like stock rl_games, confirm the resolution by hand:
#   docker run --rm --entrypoint sh <image> -c \
#       '/isaac-sim/python.sh -c "import rl_games; print(rl_games.__file__)"'
# It should print /opt/rl_games/rl_games/__init__.py, not a site-packages path.
#
# --no-deps keeps this consistent with the layers above: the fork's additions
# (algos_torch/plasticity.py, plasticity_adam.py) import nothing beyond torch
# and the stdlib, so every runtime dep is already covered by the stock install
# we are replacing. Drop it only if the fork gains a new third-party dependency.
#
# The `rm pyproject.toml` is load-bearing, not tidying. The fork carries *two*
# build configs -- a setuptools setup.py and a Poetry pyproject.toml -- and pip
# obeys the latter's [build-system]. That breaks this build twice over:
#   1. poetry-core is not installed in the Isaac Sim interpreter, so
#      --no-build-isolation dies with `ModuleNotFoundError: No module named
#      'poetry'`;
#   2. and with isolation left on, Poetry's `python = ">=3.7.1,<3.11"` becomes
#      Requires-Python, so pip refuses outright: `Package 'rl-games' requires a
#      different Python: 3.11.13 not in '<3.11,>=3.7.1'`. This image is 3.11.
# Deleting it drops pip onto the setup.py/setuptools path, which declares no
# python_requires -- and which is how rl-games is packaged on PyPI anyway, so
# nothing about the resulting install is unusual. The durable fix is to widen
# (or drop) that constraint in the fork; until then this line stays.
#
# RL_GAMES_REF defaults to the feature branch, so a plain rebuild tracks its tip
# (see the ADD below for what makes that true). Pass a full commit SHA instead --
# --build-arg RL_GAMES_REF=<sha> -- to pin a build you want to reproduce exactly.
# Either way the resolved SHA is recorded at /opt/rl_games/.build-sha, and both
# entrypoints echo it at startup, so every run's log names the commit it trained
# against. That log line is the only way back to an old run's exact fork state.
ARG RL_GAMES_REPO=https://github.com/UoA-CARES/rl_games.git
ARG RL_GAMES_REF=master

# Make the fetch below cache-correct. Docker keys a RUN layer on its command
# *text*, not on anything at the far end of the network, so `git fetch <branch>`
# is a constant string whose layer is reused forever: push to the fork and every
# machine with a warm cache keeps building the commit it first saw, silently, and
# under PCS that is per job. ADD-from-URL keys its layer on the fetched *content*
# instead, so when the branch head moves this 600-byte ref advertisement changes,
# this layer's digest changes, and everything below it is invalidated. No build
# flags involved -- which matters, because PCS just runs a plain `docker build`.
#
# This is git's own transport endpoint (it is what `git clone` requests first),
# deliberately not api.github.com: the API allows 60 unauthenticated requests per
# hour per IP, and a seed sweep building per job on a shared worker would exhaust
# that and fail the build outright on the 403.
#
# The advertisement lists *all* refs, so a push to any branch of the fork busts
# this, as can a change to GitHub's server `agent=` string. Both just re-run the
# fetch and the two small installs below (~12s total); the expensive /opt/isaaclab
# layer sits above this line and is untouched. Over-invalidation is near-free here
# by construction. Note this needs an anonymously readable HTTP remote -- point
# RL_GAMES_REPO at SSH or a private repo and ADD cannot authenticate.
ADD ${RL_GAMES_REPO}/info/refs?service=git-upload-pack /opt/rl_games.refs

# git ships in the base image, so no apt-get is needed to reach the fork.
#
# init+fetch rather than `git clone --branch`, because --branch takes branch and
# tag names only: `clone --branch <sha>` dies with "Remote branch <sha> not found
# in upstream origin", which would make the SHA pin documented above impossible.
# `fetch --depth 1 origin <ref>` accepts a branch name or a full SHA alike (GitHub
# serves arbitrary full SHAs), so one code path covers both.
RUN git init -q /opt/rl_games \
 && git -C /opt/rl_games remote add origin "${RL_GAMES_REPO}" \
 && git -C /opt/rl_games fetch -q --depth 1 origin "${RL_GAMES_REF}" \
 && git -C /opt/rl_games checkout -q FETCH_HEAD \
 && git -C /opt/rl_games rev-parse HEAD > /opt/rl_games/.build-sha \
 && rm -f /opt/rl_games/pyproject.toml \
 && /isaac-sim/python.sh -m pip install --no-cache-dir --no-build-isolation \
        --no-deps -e /opt/rl_games \
 && chmod -R o+rX /opt/rl_games

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
