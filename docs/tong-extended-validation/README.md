# TONG extended CPU validation

These are engineering checks, not the project's final experiments. The formal A/B deliverable remains **0/4 final models**. Seeds 5421 and 2026, augmentation settings and the shared training protocol still need team confirmation. The C runs below test WANG's loss integration only; they do not replace WANG's formal work.

## Observed results

- Final unit suite: **45 tests, 43 passed, 2 CUDA-only tests skipped**. Three added tests exercise actual early stopping and refused resume after stopping, AUC against independent positive-negative pair counting, and the unavailable-CUDA guard before data/weights are loaded.
- Full dataset audit: **12,000 images**, zero bad files and zero exact cross-split hash overlap. The original `Images/verification.json` was preserved. Another **100 fixture images** passed decoding, declared hashes and dimensions/format checks, with zero exact hash overlap with the formal manifest. This is not perceptual near-duplicate or content-bias auditing.
- **Eight completed real-image CPU pilot runs**, each with 32 training images, 16 validation images and two completed epochs: A/B/C at each of two provisional seeds, C with lambda=0, and A resumed after one epoch. Configurations, history, validation predictions and summaries are in the named subdirectories.
- B and C(lambda=0) matched exactly in model parameters, AdamW state, scaler state, checkpoint selection/early-stop state, epoch and history except wall-clock seconds.
- Interrupted/resumed A matched uninterrupted A in those same fields, including optimizer moments and step counts. This checks epoch-boundary resume, not mid-batch recovery.
- All **8 best checkpoints reloaded** and reproduced all **128 saved validation probabilities** exactly (maximum absolute difference 0). Each run preserved 105 frozen tensors, including every BatchNorm running-statistics buffer, while classifier weights changed.
- CLI help and Python compilation passed. No test-set model scores or final project conclusions were produced.

See [pilot-matrix.csv](pilot-matrix.csv) for the compact run inventory and [report.json](report.json) for the machine-readable evidence. Tiny-subset AUC values are diagnostic only and must not appear as formal paper/report results.

## Blocked or unverified

- **Multiprocessing:** an actual `num_workers=2` attempt failed while creating the PyTorch resource-sharer socket with `PermissionError: [Errno 1] Operation not permitted`. That attempt was stopped and is not one of the eight completed runs. Worker-0/2 training and view equality were not verified. The supported tested setting here is `num_workers=0`. See [multiprocessing-blocker.txt](multiprocessing-blocker.txt).
- **CUDA:** this cloud environment has a CPU-only PyTorch build and no available CUDA. AMP, GPU memory/throughput and GPU determinism remain unverified.
- **Formal integration:** OUYANG's formal data/view factory, frozen shared settings and target GPU integration remain outstanding. CPU is not intrinsically unable to train the full dataset, but a long run on a provisional protocol would not satisfy the final deliverable.
- **Personal computer:** only the current cloud execution environment was inspected. These results do not establish the user's own computer's capabilities.

The report deliberately sets `ok: false` for overall completion and `available_cpu_checks_passed: true` for the supported subset. Neither skipped tests nor environment-blocked attempts are counted as passes.

## Reproduction

Use the repository root and the tested environment in `requirements-tong.txt`, with access to the dataset and the official ImageNet ResNet18 weights (or the existing torch cache). The safe reproduction driver creates an unused directory under ignored `runs/`, refuses overwrites and omits the blocked multiprocessing attempt:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests docs/tong-extended-validation
python - --output-dir runs/tong_extended_validation_repeat < docs/tong-extended-validation/run_checks.py
```

The driver writes full local checkpoints and logs, requiring several GiB of free disk. Each subprocess has a five-minute timeout for this small CPU validation workload; slower hardware may need a reviewed timeout adjustment. CPU speed is not a benchmark for formal GPU scheduling.

The published driver consolidates the checks that were actually executed and removes the unsupported worker attempt. It was syntax/help checked after consolidation; the already completed eight-run matrix was not repeated merely to regenerate the same artifacts. `complete_checks.py` is the historical recovery helper used for the original interrupted run, requiring its original blocked-attempt log and fixed output directory. **It is not a second step for new reproductions.**

## Provenance and scope

The actual training ran against clean source commit `a337e000aa81ebaa60884c27c53870b5fa7fd448`. The initial matrix's unit-test command ran the existing 42-test suite (40 passed, 2 CUDA skips); the final `unit-tests.txt` records the expanded 45-test suite. The new tests and evidence documents do not alter model or training behavior.

`tested-source-sha256.json` lists the tested source/test file hashes, including the new regression test. `report.json` retains the observed training commit and adds the final-suite result. Checkpoint SHA256 values identify local files; they are not download links. Checkpoints, ImageNet weights, caches and image data were not added by this update. Only small configs, CSV/JSON evidence, logs, helper scripts, tests and the [team handoff checklist](../tong-team-handoff.md) are published.
