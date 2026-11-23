#!/bin/bash
#SBATCH --job-name=tong-ab-formal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this template through your authorized Slurm GPU allocation}"
project=${1:?Pass your own absolute project checkout}
commit=${2:?Pass the full reviewed Git commit}
qa=${3:?Pass your fresh full QA JSON path}
ready=${4:?Pass the matching readiness JSON path}
runs_root=${5:-runs/tong_formal_ab}
module load Anaconda3/2023.09-0
cd "$project"
source "${TONG_VENV:-$PWD/.venv}/bin/activate"
export TORCH_HOME="${TORCH_HOME:-$PWD/runs/torch-cache}"
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m tools.training.run_formal_ab --execute --expected-commit "$commit" \
  --qa "$qa" --data-ready "$ready" --runs-root "$runs_root"
