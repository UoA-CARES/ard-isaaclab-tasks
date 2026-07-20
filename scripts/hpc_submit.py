#!/usr/bin/env python
"""Submit ARD IsaacLab training jobs to the CARES HPC Scheduler.

Wraps the `hpc_client` Python API (https://uoa-cares.github.io/hpc-client/) so a
seed sweep is one command instead of one hand-written job.json per run:

    python scripts/hpc_submit.py --task Isaac-ARD-Cartpole-v0 --seeds 0 1 2

Every job runs the same prebuilt image and differs only in the `command`, which
carries the training flags to scripts/hpc_entrypoint.sh. Configuration travels in
`command` (not in the job `env` block) because the CARES HPC Scheduler does not
inject a job's `env` into the container -- anything put there is silently dropped,
so the entrypoint would fall back to its defaults (Isaac-ARD-Cartpole-v0, no seed).
The image must already be in the registry -- push it with scripts/hpc_push.sh first.

Prerequisites (one-off):

    pip install -e /home/lee/hpc-client        # provides the hpc_client package
    hpc-client configure --scheduler-url http://<scheduler-host>:8080
    hpc-client login <upi>

Note: the published docs show a `submit_job`/`wait_for_job`/`get_logs` API, but
the shipped client exposes `submit`/`wait`/`logs`. This script uses the latter.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from typing import Any

try:
    from hpc_client import HPCClient
    from hpc_client.config import HPCConfig
    from hpc_client.errors import HPCAuthenticationError, HPCClientError
except ImportError:
    sys.exit(
        "hpc_client is not installed. Install the HPC client into this "
        "interpreter, e.g.:\n    pip install -e /home/lee/hpc-client"
    )

DEFAULT_IMAGE = os.environ.get(
    "ARD_HPC_IMAGE", "130.216.238.2:5500/cli797_ard-isaaclab:2.3.2"
)
ENTRYPOINT = "bash /opt/ard-isaaclab-tasks/scripts/hpc_entrypoint.sh"

# The scheduler rejects submissions once a user has this many active jobs.
MAX_ACTIVE_JOBS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit ARD IsaacLab jobs to the CARES HPC Scheduler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default="Isaac-ARD-Cartpole-v0", help="Task id.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Registry image to run.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Submit one job per seed. Omit for a single job with the task default.",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="--num_envs.")
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="--max_iterations."
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=2.0,
        help="Job is killed past this. Overestimate rather than under.",
    )
    parser.add_argument(
        "--extra-args", default="", help="Raw args appended verbatim to train.py."
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra container env var; repeatable (e.g. --env WANDB_API_KEY=...).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME",
        help="NAS dataset to mount read-only at /workspace/datasets/NAME; repeatable.",
    )
    parser.add_argument("--name-prefix", default="ard", help="Prefix for job names.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job specs that would be submitted and exit.",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Block until every submitted job finishes."
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Print each job's logs once it finishes (implies --wait).",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build one job spec per seed (or a single job when no seeds are given)."""
    extra_args = args.extra_args

    # The vision benchmark instantiates a TiledCamera, which train.py only allows
    # with --enable_cameras. Forgetting it fails the job minutes in, on a worker,
    # for a reason the logs make you dig for -- so add it here instead.
    if "Vision" in args.task and "--enable_cameras" not in extra_args:
        extra_args = f"{extra_args} --enable_cameras".strip()
        print(f"[hpc] vision task: added --enable_cameras -> EXTRA_ARGS={extra_args!r}")

    # The scheduler drops the job `env` block, so it holds only user-supplied
    # extras (--env). They are best-effort: if the scheduler ignores env, these
    # are ignored too -- hence the warning. Secrets like WANDB_API_KEY that must
    # reach the container therefore can't be relied on here.
    base_env: dict[str, str] = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator:
            sys.exit(f"--env expects KEY=VALUE, got: {item!r}")
        base_env[key] = value
    if base_env:
        print(
            "[hpc] warning: the HPC scheduler does not inject the job `env` block "
            "into the container, so these --env values (e.g. WANDB_API_KEY) may be "
            "ignored: " + ", ".join(sorted(base_env))
        )

    # Flags common to every job in the sweep, forwarded verbatim to train.py by
    # hpc_entrypoint.sh via "$@". (--headless is added by the entrypoint.)
    base_flags = ["--task", args.task]
    if args.num_envs is not None:
        base_flags += ["--num_envs", str(args.num_envs)]
    if args.max_iterations is not None:
        base_flags += ["--max_iterations", str(args.max_iterations)]
    if extra_args:
        base_flags += shlex.split(extra_args)

    # Job names are how you find a run in `hpc-client jobs`, so make them say what
    # the run is: ard_cartpole_seed_1 rather than job_3.
    slug = args.task.removeprefix("Isaac-ARD-").removesuffix("-v0").lower()
    slug = slug.replace("-", "_")

    seeds = args.seeds if args.seeds is not None else [None]
    jobs = []

    for seed in seeds:
        flags = list(base_flags)
        name = f"{args.name_prefix}_{slug}"

        if seed is not None:
            flags += ["--seed", str(seed)]
            name = f"{name}_seed_{seed}"

        command = f"{ENTRYPOINT} " + " ".join(shlex.quote(flag) for flag in flags)

        jobs.append(
            {
                "job_name": name,
                "image": args.image,
                "command": command,
                "max_runtime_hours": args.max_runtime_hours,
                "required_datasets": list(args.dataset),
                "required_worker_ids": [],
                "env": base_env,
            }
        )

    return jobs


def job_id_of(response: Any) -> str:
    """Pull the job id out of a submit response (str or dict, depending on server)."""
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        for key in ("job_id", "id"):
            if key in response:
                return str(response[key])

        job = response.get("job")
        if isinstance(job, dict) and "job_id" in job:
            return str(job["job_id"])

    return str(response)


def connect() -> HPCClient:
    """Load the stored session, re-logging in with $HPC_PASSWORD if it has expired."""
    try:
        client = HPCClient.from_config()
    except RuntimeError as error:
        sys.exit(
            f"{error}\nConfigure the client first:\n"
            "    hpc-client configure --scheduler-url http://<scheduler-host>:8080\n"
            "    hpc-client login <upi>"
        )

    try:
        client.me()
        return client
    except HPCAuthenticationError:
        pass

    config = HPCConfig.load()
    password = os.environ.get("HPC_PASSWORD")

    if not (config.username and password):
        sys.exit(
            "HPC session expired. Run `hpc-client login <upi>`, or set "
            "$HPC_PASSWORD to have this script re-login automatically."
        )

    print(f"[hpc] session expired; re-logging in as {config.username}")
    client.login(username=config.username, password=password)
    return client


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)

    if args.dry_run:
        import json

        print(json.dumps(jobs, indent=2))
        return

    client = connect()

    active = len(client.jobs())
    if active + len(jobs) > MAX_ACTIVE_JOBS:
        sys.exit(
            f"{active} active job(s) + {len(jobs)} new would exceed the "
            f"{MAX_ACTIVE_JOBS}-job limit. Submit a smaller batch, or wait for "
            "running jobs to finish."
        )

    job_ids = []
    for job in jobs:
        try:
            response = client.submit(job)
        except HPCClientError as error:
            sys.exit(f"submit failed for {job['job_name']}: {error}")

        job_id = job_id_of(response)
        job_ids.append(job_id)
        print(f"[hpc] submitted {job['job_name']}: {job_id}")

    print(f"\n[hpc] {len(job_ids)} job(s) queued. Track them with:")
    print("      hpc-client jobs")
    print(f"      hpc-client logs {job_ids[0]}")
    print("      outputs land on the NAS at /cares-nas/hpc/outputs/<upi>/<job_id>")

    if not (args.wait or args.logs):
        return

    for job_id in job_ids:
        print(f"\n[hpc] waiting for {job_id} ...")
        data = client.wait(job_id)
        job = data.get("job", data)
        print(f"[hpc] {job_id}: {job.get('status')}")

        if args.logs:
            logs = client.logs(job_id)
            print(logs.get("container_log", "(no container log)"))


if __name__ == "__main__":
    main()
