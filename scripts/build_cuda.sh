#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NIRNAY - configure and build the GPU-enabled solver.
#
#   scripts/build_cuda.sh                      # build-cuda/, Release, the default architectures
#   scripts/build_cuda.sh build-b200 100       # a B200-only build in build-b200/
#   CUDA_HOME=/opt/cuda-12.8 scripts/build_cuda.sh
#
# Finds nvcc on PATH, under $CUDA_HOME, under /usr/local/cuda*, or in a conda environment
# ($NIRNAY_CUDA_ENV, default "nirnay") that has the cuda-nvcc and libcusparse-dev packages,
# which is how a cluster with a driver but no system toolkit gets one:
#
#   conda create -n nirnay -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev \
#                          cuda-cccl libcusparse-dev cmake ninja
#
# The build runs in a CLEAN environment. Activating a conda environment that carries a
# compiler package exports CFLAGS/CXXFLAGS/LDFLAGS pointing at conda's sysroot, and linking
# the system compiler's objects against that sysroot fails in librt (__pthread_attr_copy).
# Only PATH is borrowed, for cmake, ninja and nvcc.
#
# The toolkit must not be newer than the driver: `nvidia-smi` prints the highest CUDA
# version the driver supports. Blackwell (B200, sm_100) needs 12.8 or newer.
set -euo pipefail

BUILD_DIR="${1:-build-cuda}"
ARCHITECTURES="${2:-80;90;100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

find_nvcc() {
  local env_name="${NIRNAY_CUDA_ENV:-nirnay}"
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
  local candidates=(
    "$(command -v nvcc 2>/dev/null || true)"
    "${CUDA_HOME:-/nonexistent}/bin/nvcc"
    /usr/local/cuda/bin/nvcc
    /usr/local/cuda-*/bin/nvcc
    "$conda_base/envs/$env_name/bin/nvcc"
  )
  for nvcc in "${candidates[@]}"; do
    if [ -n "$nvcc" ] && [ -x "$nvcc" ]; then
      echo "$nvcc"
      return 0
    fi
  done
  return 1
}

NVCC="$(find_nvcc)" || {
  echo "error: no nvcc found. Install a CUDA toolkit (12.8+ for B200), set CUDA_HOME, or" >&2
  echo "       create the conda environment described at the top of this script." >&2
  exit 1
}
TOOLKIT="$(cd "$(dirname "$NVCC")/.." && pwd)"
HOST_CXX="$(command -v g++-13 || command -v g++)"
echo "nirnay: nvcc ${NVCC} ($("$NVCC" --version | tail -1)), host compiler ${HOST_CXX}"

env -i HOME="$HOME" PATH="$(dirname "$NVCC"):/usr/local/bin:/usr/bin:/bin" \
  CUDACXX="$NVCC" bash -c "
    set -euo pipefail
    cd '$ROOT'
    scripts/configure.sh '$BUILD_DIR' Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DNIRNAY_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES='$ARCHITECTURES' \
      -DCUDAToolkit_ROOT='$TOOLKIT' \
      -DCMAKE_CUDA_HOST_COMPILER='$HOST_CXX'
    cmake --build '$BUILD_DIR' -j \"\$(nproc)\"
  "
echo "nirnay: built ${BUILD_DIR}/nirnay with the CUDA backend; run with --gpu"
