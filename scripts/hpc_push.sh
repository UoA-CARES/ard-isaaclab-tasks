#!/usr/bin/env bash
# Build the ARD IsaacLab image and push it to the CARES registry, so the HPC
# Scheduler can pull it. Run from the repo root:
#
#   scripts/hpc_push.sh                       # build + push the default tag
#   IMAGE=130.216.238.2:5500/cli797_ard-isaaclab:dev scripts/hpc_push.sh
#   scripts/hpc_push.sh --no-build            # push an already-built image
#
# The scheduler pulls the image, so a reward edit only reaches the cluster after
# this script has run. The rebuild is incremental: only the `ard_tasks` source
# layer and the editable-install layer change, so a re-push after a reward edit
# uploads a few MB, not the 17 GB base.
set -euo pipefail

REGISTRY="${REGISTRY:-130.216.238.2:5500}"
IMAGE="${IMAGE:-${REGISTRY}/cli797_ard-isaaclab:2.3.2}"
BUILD=1

for arg in "$@"; do
  case "$arg" in
    --no-build) BUILD=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# The CARES registry is plain HTTP, so Docker refuses to talk to it until it is
# listed as an insecure registry. Fail early with the exact fix rather than
# letting `docker push` die with an opaque TLS error.
if ! docker info 2>/dev/null | grep -qF "$REGISTRY"; then
  cat >&2 <<EOF
[hpc] '$REGISTRY' is not in Docker's insecure-registries list, so the push
      would fail with an HTTPS/TLS error. Add it once per machine:

        sudo tee -a /etc/docker/daemon.json  # merge this key into the file:
        { "insecure-registries": ["$REGISTRY"] }

        sudo systemctl restart docker

      Then verify:  curl -s http://$REGISTRY/v2/_catalog
EOF
  exit 1
fi

if [ "$BUILD" -eq 1 ]; then
  echo "[hpc] building $IMAGE"
  docker build -t "$IMAGE" .
fi

echo "[hpc] pushing $IMAGE"
docker push "$IMAGE"

echo "[hpc] done. Submit with:"
echo "      python scripts/hpc_submit.py --task Isaac-ARD-Cartpole-v0 --image $IMAGE"
