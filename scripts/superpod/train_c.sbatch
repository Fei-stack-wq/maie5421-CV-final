#!/bin/bash
#SBATCH --job-name=wang-c-formal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
set -euo pipefail
module load Anaconda3/2023.09-0
cd "${1:?Pass the absolute project directory}"
source .venv/bin/activate
export TORCH_HOME="$PWD/runs/torch-cache"
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import json; assert json.load(open("runs/superpod-data-ready.json"))["ready_for_training"]'
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > runs/superpod-gpu.csv
python -m pytest -q --junitxml=runs/superpod-gpu-tests.xml
python -m tools.training.cuda_pilot --config examples/wang/pilot_c_ouyang_cuda.yaml --output-dir "runs/superpod-pilot-${SLURM_JOB_ID}"
for seed in 5421 2026; do
  python -m src.train --config "configs/wang/c_${seed}.yaml" --output-dir "runs/C_${seed}"
done
echo 'Both formal C runs completed'
