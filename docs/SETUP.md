# Setup

## Tested runtime

The winning run used Python 3.11.13, Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`, Isaac Lab revision `42e61645c96bac08135566634785cdc87728d5ab` (2.3.2), Torch `2.7.0+cu128`, RSL RL 3.1.2, tensordict 0.11.0 and h5py 3.15.1. It ran on one RTX 5090 with 32607 MiB reported memory. See the [runtime receipt](../results/hammer_best/manifest.json).

With the existing Isaac image, use its bundled Python and a checkout of the pinned Lab revision. `scripts/supermimic/python.sh` adds the Lab and repository source directories to `PYTHONPATH` without replacing the shared image's packages.

```bash
export CONTRACK_ISAACLAB_PATH=/absolute/path/to/IsaacLab
export CONTRACK_RESULTS_DIR=/absolute/path/to/experiment-results
# Optional if Isaac is not installed at /isaac-sim:
export ISAAC_SIM_PYTHON=/absolute/path/to/isaac-sim/python.sh
bash scripts/supermimic/python.sh -c 'import torch, isaaclab; print(torch.__version__)'
bash scripts/supermimic/python.sh -m pip install --no-deps -e .
```

When Lab is already installed into the selected Python environment, `CONTRACK_ISAACLAB_PATH` is optional. When `ISAAC_SIM_PYTHON` is unset, the launcher uses `/isaac-sim/python.sh` if available, otherwise the active `python3`. Results default to the ignored `outputs/` directory; an absolute `CONTRACK_RESULTS_DIR` is recommended.

## Fresh installation

Use Python 3.11 in a dedicated environment. Follow NVIDIA's Isaac Sim installation requirements for your machine.

```bash
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git /absolute/path/to/IsaacLab
cd /absolute/path/to/IsaacLab
git checkout 42e61645c96bac08135566634785cdc87728d5ab
python -m pip install 'setuptools<82' wheel
python -m pip install --no-build-isolation flatdict==4.0.1
./isaaclab.sh --install rsl-rl
cd /absolute/path/to/Supermimic-dex
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e source/ConTrack
```

`requirements.txt` pins the recorded image's auxiliary Python packages. Torch/Isaac are installed separately. For video recording, install `python -m pip install -e '.[video]'` in the Isaac interpreter if those packages are absent.

## Resources and generated assets

```bash
python3 scripts/supermimic/fetch_assets.py
python3 scripts/supermimic/fetch_assets.py --verify-only
bash scripts/supermimic/python.sh scripts/supermimic/prepare_assets.py --device cpu
```

The downloader retrieves robot URDFs/meshes, the Hammer clip and two Cylinder defaults needed by upstream registration. It verifies SHA256 and byte size for each file from the pinned ConTrack archive. Other upstream clips must be supplied separately when using the optional multi-task suite utilities.

Downloaded resources stay in ignored `assets/meshes`, `assets/urdf` and `data` directories. Converted USD, collision meshes, logs and compiler caches go under `CONTRACK_RESULTS_DIR` through ignored symlinks. Use one results root for one resource set; a different output-link target is rejected rather than silently reused.

The original run's generated USD hashes are retained in the manifest. Converting resources on a different native simulator build can change these files; the source-resource hashes alone do not guarantee identical generated USD or PhysX trajectories.

Rendering additionally needs an operational RTX/Vulkan graphics stack. CUDA-only access is sufficient for some checks but does not establish that native rendering works. A headless GPU render job can run the README's `record_video.py` command when the interactive container lacks a usable graphics context.
