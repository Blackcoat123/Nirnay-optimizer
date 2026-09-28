#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NIRNAY - the GPU evidence runs on one B200, as a Slurm batch job.
#
#   sbatch bench/slurm/b200_evidence.sbatch                      # a full B200
#   sbatch -p small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 \
#          --export=ALL,NIRNAY_MACHINE_TAG=dgx-b200-mig1g45 bench/slurm/b200_evidence.sbatch
#
# Build first on the login node (compiling needs no GPU):  scripts/build_cuda.sh build-cuda 100
# The machine tag goes into every CSV row: name the slice precisely when it is not a full card,
# because a MIG slice has a fraction of the card's bandwidth and the numbers must say so.
#SBATCH --job-name=nirnay-evidence
#SBATCH --partition=full-b200
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=bench/results/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
export NIRNAY_MACHINE_TAG="${NIRNAY_MACHINE_TAG:-dgx-b200-full}"
PY="${PYTHON:-python3}"
BIN="${NIRNAY_BINARY:-build-cuda/nirnay}"
TAG="$(echo "$NIRNAY_MACHINE_TAG" | tr -c 'A-Za-z0-9.\n-' '-')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Scale, against optima known exactly by construction: PDHG on the GPU.
for family in refinery random staircase; do
  case "$family" in
    refinery) sizes="12 365 8760" ;;
    random) sizes="1000 5000 20000 100000 1000000" ;;
    staircase) sizes="1000 5000 20000 100000" ;;
  esac
  "$PY" bench/runners/scale.py --binary "$BIN" --structure "$family" --sizes $sizes \
    --engines pdhg --solver-option gpu=true --time-limit 600 --keep bench/large \
    --out "bench/results/scale-${family}-gpu-${TAG}.csv"
done

# The standard sets. auto races the engines on anything of 1,000 rows or more, with the GPU
# as the first-order racer.
"$PY" bench/runners/netlib.py --binary "$BIN" --time-limit 120 \
  --out "bench/results/netlib-full-${TAG}.csv" || true
"$PY" bench/runners/mittelmann.py --binary "$BIN" --time-limit 300 \
  --out "bench/results/mittelmann-${TAG}.csv" || true
"$PY" bench/runners/miplib.py --binary "$BIN" --time-limit 60 \
  --out "bench/results/miplib-${TAG}.csv" || true
"$PY" bench/runners/maros_meszaros.py --binary "$BIN" --time-limit 60 \
  --out "bench/results/maros-meszaros-${TAG}.csv" || true
