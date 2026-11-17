# TONG: formal A/B preparation and handoff

**Preparation only. Four formal A/B models are still NOT RUN (0/4).** This work
starts from WANG integration commit `0cb7e982c91e6be208dc3d18afc584988f5f0f5f`.
It adds separate files; it does not change the trainer, OUYANG data pipeline,
completed C experiments, `main`, or a teammate's branch.

## Four configurations

| Run | Condition | Seed | Configuration |
| --- | --- | --- | --- |
| A_5421 | clean-view BCE | 5421 | `configs/tong/a_5421.yaml` |
| B_5421 | mean paired-view BCE | 5421 | `configs/tong/b_5421.yaml` |
| A_2026 | clean-view BCE | 2026 | `configs/tong/a_2026.yaml` |
| B_2026 | mean paired-view BCE | 2026 | `configs/tong/b_2026.yaml` |

Every configuration matches the same seed's committed C configuration except
condition and consistency weight (0 for A/B, 1 for C). Shared settings are
ImageNet ResNet-18, train layer4/head with frozen BN statistics, OUYANG v1,
AdamW 1e-4/0.01, microbatch 4 × accumulation 4, at most 8 epochs,
clean-validation AUC selection, patience 2, deterministic execution and CUDA AMP
initial scale 1024. See [the agreed protocol](wang-training-protocol.md).
`protocol_confirmed: true` means the settings were agreed, not that training ran.

## Resources and authorization

Use **your own authorized** GPU allocation, project checkout and environment.
No account, password, allocation identifier or teammate's directory is supplied.
The Slurm file is an optional template, not a submitted job. Review the site's
partition, module, memory/time limits and your account/QOS before submitting it.
Do not train on a login node.

- One CUDA GPU; the completed C reference used one H800. Prefer matching hardware
  for comparison. This preparation has not established a minimum VRAM or runtime
  for all four A/B experiments on other GPUs.
- Python 3.11; `torch==2.3.0` and `torchvision==0.18.0` with cu121;
  remaining versions in `requirements-training.txt`. The runner checks these.
- All 12,000 manifest images plus the 100-image fixture, fresh complete
  byte/decode/pHash QA and its readiness report. No subset qualifies as formal.
- Verified official ImageNet weights already cached under your `TORCH_HOME`.
- Budget at least 20 GB free space as a conservative planning allowance for data,
  environment, caches and multiple best-epoch/last checkpoints. This is not a
  measured minimum. Keep large weights and raw outputs out of Git.

Current cloud preparation/testing used CPU and Python 3.12. That is useful for
unit tests, **not** evidence of formal CUDA training or the user's own computer's
capabilities. Neither GPU jobs nor formal training were started here.

## Prepare your checkout

Use a reviewed fixed commit containing these new files and a full checkout of
`Images/` metadata and image files. Do not reuse or overwrite another person's
checkout. The existing [data preparation workflow](../scripts/superpod/README.md)
and `scripts/superpod/prepare.sbatch` perform environment/data/weights preparation
only; despite the historical `wang-c-prepare` job name, that preparation script
does not train C. Review it before use, and run it only in your own fresh
workspace and authorized CPU allocation. Its output paths are:

- `runs/superpod-qa.json`: fresh full QA;
- `runs/superpod-data-ready.json`: validated readiness;
- `runs/torch-cache`: official pretrained weights.

The QA screen may exit 3 for already documented false-positive candidates. Only
`tools.training.check_received_qa` may accept candidates matching the versioned
review. New/changed candidates, missing files or other errors must stop the work;
do not bypass that gate. Preserve raw QA and existing human-review records.

## Plan without training (default)

From the repository root with dependencies installed:

```bash
python -m tools.training.run_formal_ab
```

This prints the four configurations/commands. It does not create outputs, check
or allocate a GPU, download anything, or start a subprocess. A chosen run output
must be empty or absent; existing results and symlinks are rejected.

## Execute only inside your authorized GPU allocation

After review, explicitly record the full 40-character source commit. The runner
requires a clean checkout at that exact commit. Do not substitute an unreviewed
moving branch name. Example inside an allocation you already control:

```bash
export TORCH_HOME="$PWD/runs/torch-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REVIEWED_COMMIT=<full-reviewed-commit-sha>
python -m tools.training.run_formal_ab --execute \
  --expected-commit "$REVIEWED_COMMIT" \
  --qa "$PWD/runs/superpod-qa.json" \
  --data-ready "$PWD/runs/superpod-data-ready.json"
```

Replace the angle-bracket placeholder; it is not a runnable value. For an
appropriate Slurm installation, `scripts/tong/formal_ab.sbatch` wraps this same
command. Pass your absolute project directory, reviewed commit, absolute QA
path and absolute readiness path as its four arguments; supply your own allowed
account/QOS and output log path when submitting. For example, after replacing all placeholders:

```bash
mkdir -p runs
sbatch --account=<your-authorized-account> --qos=<your-authorized-gpu-qos> \
  --output="$PWD/runs/formal-ab-%j.out" scripts/tong/formal_ab.sbatch \
  "$PWD" "$REVIEWED_COMMIT" "$PWD/runs/superpod-qa.json" \
  "$PWD/runs/superpod-data-ready.json"
```

The explicit log path keeps Slurm output out of the tracked source tree; the
runner rejects a dirty checkout. No submission is made by this repository or by
merely printing the plan.

Before any training process is launched, execution checks clean source,
CUDA/runtime/dependencies, cached weight hash, fresh QA/readiness linkage, reviewed
data version and actual hashes of every manifest image. Then each selected run
trains and independently reloads its best checkpoint against all 1,500 clean
validation images. A failed training or verification stops later runs.

Model outputs: `runs/tong_formal_ab/<run_id>/`.
Separate attempt receipts/logs: `runs/tong_formal_ab-attempts/<unique_attempt>/`.
Reports are outside the model directory tree and never overwrite old reports.
`verified_models` counts only successful reload checks, not completed processes.

To execute only an **unstarted** subset, repeat `--run-id`, for example
`--run-id A_2026 --run-id B_2026`. Existing outputs are not silently retrained or
resumed. Inspect failures first, preserve their outputs, and agree any repair or
new run explicitly. The inherited trainer has a known interruption boundary:
`last.pt` can exist before the final CSV/summary is written, and a final-epoch
resume will not rebuild those missing files. The verifier intentionally rejects
such incomplete artifacts rather than declaring success.

## Independent reload / downstream handoff

With all data available, verification can run on CPU without retraining:

```bash
python -m tools.training.verify_formal_ab \
  --runs-root runs/tong_formal_ab --device cpu \
  --output runs/tong-ab-verification.json
```

The report must be new and outside the input run directory tree. Use repeated
`--run-id` for available subsets; the report distinguishes `selected_verified`
from `all_four_verified`. The verifier checks protocol, full data provenance,
clean source revision, epoch/update/early-stop records, best/last checkpoint
agreement, validation IDs/labels/probabilities, and reloaded metrics/probabilities
within 1e-5. It records SHA256 for all handoff files. A zero-update epoch is rejected. Any
nonzero AMP skip count is marked `budget_review_required` because the completed
C runs had no skips; reload success alone does not settle effective-budget
comparability. Inspect that flag before accepting the experiments for the paper.
The check of `last.pt` covers records and hashes only, not model/optimizer/RNG
resume execution. Only `best.pt` is independently loaded for inference. CPU/GPU numerical differences
beyond tolerance require investigation, not automatic relaxation.

After real training and successful reload, hand off each `best.pt`, `config.yaml`,
`history.csv`, `summary.json`, `val_predictions.csv`, verification report and
hashes. Preserve `last.pt` and best-epoch checkpoints for recovery. Use an agreed
large-file delivery location, not Git, for model weights. ZHANG owns held-out
robustness evaluation; ZHUO owns inference/CLI integration. This work does not
produce final-test scores, confidence intervals, plots or those teammates' work.

Before claiming completion, fill the actual-run fields in
[the experimental-setup draft](tong-experimental-setup.md), review all four runs,
and coordinate checkpoints/hashes and loading interface with both recipients.
