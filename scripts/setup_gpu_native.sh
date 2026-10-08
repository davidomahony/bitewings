#!/usr/bin/env bash
# Install the GPU training environment directly on a CUDA machine, without Docker.
#
# For container-based GPU rentals (e.g. Vast.ai, RunPod) where `docker build` is not
# available. Mirrors docker/Dockerfile.gpu step for step. Start the instance from
#     nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
# (the devel variant: compiling onedl-mmcv's CUDA ops needs nvcc), then from the repo root:
#     bash scripts/setup_gpu_native.sh
#     source /opt/venv/bin/activate
#
# Re-running is safe: finished steps are skipped. Environment overrides:
#     TORCH_CUDA_ARCH_LIST  compute capability to compile for (default: detected from nvidia-smi)
#     VENV                  virtualenv location (default /opt/venv)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:-/opt/venv}"
log() { echo -e "\n=== $*"; }

log "GPU and driver"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';')"
fi
export TORCH_CUDA_ARCH_LIST
echo "compiling CUDA ops for: $TORCH_CUDA_ARCH_LIST"
command -v nvcc >/dev/null || { echo "nvcc not found: use a *-devel* CUDA image"; exit 1; }
nvcc --version | tail -1

if [ ! -f onedl-mmdetection/setup.py ]; then
    log "Fetching the onedl-mmdetection submodule"
    git submodule update --init --recursive
fi

log "System packages"
if ! dpkg -s python3-venv ninja-build libgl1 >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv git build-essential ninja-build libgl1 libglib2.0-0 curl unzip
fi

log "Python virtualenv at $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

log "PyTorch (CUDA 12.9 build) and requirements"
if ! python -c "import torch, sys; sys.exit(not torch.version.cuda)" 2>/dev/null; then
    pip install --no-cache-dir setuptools wheel -r pip_requirements.txt
fi

log "onedl-mmengine and onedl-mmcv (compiles CUDA ops, ~20-40 min)"
if ! python -c "import mmcv.ops, sys; from mmcv.ops import MultiScaleDeformableAttention" 2>/dev/null; then
    export MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-$(nproc)}"
    pip install --no-cache-dir --no-build-isolation -r mim_requirements.txt
fi

log "onedl-mmdetection (editable)"
pip install --no-cache-dir -e onedl-mmdetection

log "Verifying the GPU stack"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU (driver too old for CUDA 12.9?)'
major, minor = torch.cuda.get_device_capability()
arch = f'sm_{major}{minor}'
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(),
      '| supported archs', torch.cuda.get_arch_list())
assert arch in torch.cuda.get_arch_list(), f'this torch build does not support {arch}'

# the compiled multi-scale deformable attention op must run on the GPU and agree
# with mmcv's pure-PyTorch reference implementation
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
torch.manual_seed(0)
bs, heads, dim, levels, points, queries = 1, 2, 8, 2, 2, 5
shapes = torch.tensor([[6, 4], [3, 2]], device='cuda')
starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
value = torch.rand(bs, int(shapes.prod(1).sum()), heads, dim, device='cuda')
locs = torch.rand(bs, queries, heads, levels, points, 2, device='cuda')
weights = torch.rand(bs, queries, heads, levels, points, device='cuda')
weights = weights / weights.sum((-1, -2), keepdim=True)
cuda_out = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, locs, weights, 64)
ref_out = multi_scale_deformable_attn_pytorch(value, shapes, locs, weights)
assert torch.allclose(cuda_out, ref_out, atol=1e-4), (cuda_out - ref_out).abs().max()
print('mmcv CUDA deformable attention: OK (matches the PyTorch reference)')

import mmdet, mmengine, mmcv
print('mmdet', mmdet.__version__, '| mmengine', mmengine.__version__, '| mmcv', mmcv.__version__)
EOF

log "Done. Activate with: source $VENV/bin/activate"
