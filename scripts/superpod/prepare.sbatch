#!/bin/bash
#SBATCH --job-name=wang-c-prepare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
module load Anaconda3/2023.09-0
cd "${1:?Pass the absolute project directory}"
mkdir -p runs
python -m venv .venv
source .venv/bin/activate
python -m pip install --disable-pip-version-check torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --disable-pip-version-check -r requirements-training.txt
python -m tools.dataset.receive_images --workers 8 --output runs/superpod-receipt.json
set +e
python -m tools.dataset.qa_received --check-images --phash --output runs/superpod-qa.json
qa_status=$?
set -e
if [[ "$qa_status" != 0 && "$qa_status" != 3 ]]; then exit "$qa_status"; fi
python -m tools.training.check_received_qa --qa runs/superpod-qa.json --output runs/superpod-data-ready.json
export TORCH_HOME="$PWD/runs/torch-cache"
python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
python -m pytest -q --junitxml=runs/superpod-cpu-tests.xml
python -m pip freeze > runs/superpod-environment.txt
echo 'Preparation finished: data, environment, weights and CPU tests verified'
