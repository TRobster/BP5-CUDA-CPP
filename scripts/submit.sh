#!/bin/bash
# Build and run bp5CUDA on a GPU node.
#
#   sbatch scripts/submit.sh            # default N from the .dat (128)
#   sbatch scripts/submit.sh 32         # override N
#
#SBATCH --job-name=bp5-assemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=bp5-%j.out

set -euo pipefail

module purge
module load gcc/13.1.0 eigen/3.4.0 cuda/13.0

# cuDSS does not ship with the cuda module here. Point CUDSS_ROOT at your own
# install (https://developer.nvidia.com/cudss) or leave it unset to run the
# host-only path.
: "${CUDSS_ROOT:=$HOME/opt/cudss}"

cd "$SLURM_SUBMIT_DIR"

N="${1:-}"
ARGS=(../Thrase.jl/examples/bp5-qd.dat)
[[ -n "$N" ]] && ARGS+=(--n "$N")

if [[ -d "$CUDSS_ROOT" ]]; then
  echo "building with cuDSS from $CUDSS_ROOT"
  make clean
  make -j8 CUDSS=1 CUDSS_ROOT="$CUDSS_ROOT" CUDA_HOME="$CUDA_HOME" ARCH=sm_80
  ARGS+=(--backend cudss)
else
  echo "CUDSS_ROOT=$CUDSS_ROOT not found; building host-only path"
  make clean
  make -j8
  ARGS+=(--backend eigen)
fi

nvidia-smi || true
./bp5_assemble "${ARGS[@]}"
