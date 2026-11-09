#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
lab_dir=${CONTRACK_ISAACLAB_PATH:-${ISAACLAB_PATH:-}}
export CONTRACK_RESULTS_DIR=${CONTRACK_RESULTS_DIR:-"$repo_dir/outputs"}
results_dir=$CONTRACK_RESULTS_DIR
mkdir -p "$results_dir/cache/torchinductor" "$results_dir/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$results_dir/cache/torchinductor"
export TRITON_CACHE_DIR="$results_dir/cache/triton"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONPATH="$repo_dir:$repo_dir/source/ConTrack:${PYTHONPATH:-}"
if [[ -n "$lab_dir" ]]; then
    if [[ ! -d "$lab_dir/source/isaaclab" ]]; then
        printf 'Invalid CONTRACK_ISAACLAB_PATH: %s\n' "$lab_dir" >&2
        exit 1
    fi
    export PYTHONPATH="$lab_dir/source/isaaclab:$lab_dir/source/isaaclab_rl:$lab_dir/source/isaaclab_tasks:$lab_dir/source/isaaclab_assets:$PYTHONPATH"
fi
if [[ -n "${ISAAC_SIM_PYTHON:-}" ]]; then
    exec "$ISAAC_SIM_PYTHON" "$@"
elif [[ -x /isaac-sim/python.sh ]]; then
    exec /isaac-sim/python.sh "$@"
else
    exec python3 "$@"
fi
