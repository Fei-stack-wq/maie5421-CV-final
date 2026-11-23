# CPU workflow validation

These are small workflow checks, not the formal A/B experiment results. Each run used 32 training images and 16 clean validation images, balanced by label, one epoch, seed 5421, ImageNet weights, and two optimizer updates. Do not infer comparative model quality from these tiny pilots.

Training was executed on the clean local code commit recorded in each summary. Later local commits add only validation evidence. Browser upload may consolidate the local commits into one remote commit; source-sha256.json records the exact tested source files for content verification. No held-out test split was scored.

From the repository root, with `requirements-tong.txt` installed:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m src.train --config examples/tong/pilot_a.yaml --output-dir runs/tong_verified_A_5421
python -m src.train --config examples/tong/pilot_b.yaml --output-dir runs/tong_verified_B_5421
python - < docs/tong-validation/verify_pilots.py
```

Use empty output directories; the trainer will refuse to overwrite these runs. Download/cache access is required for the first ImageNet-weight load. The reload check uses only each saved checkpoint, without downloading weights.

- `A/` and `B/`: full resolved config, per-epoch log, best-model predictions, run summary and reload check copied from the actual run.
- `unit-tests.txt`: 42 tests, 40 passed and 2 CUDA checks skipped. The extra printed epoch metrics are synthetic unit-test fixtures, not project experiment results.
- `verify_pilots.py`: reads the two runs above, reloads best.pt, re-scores clean validation, and requires exact equality with saved probabilities. It also records the checkpoint SHA256 and size.

Both pilots verified all 12,000 manifest SHA256 entries, used real repository images, and reloaded all 16 validation predictions with maximum absolute probability difference 0.0. AUC was 0.765625 for A and 0.734375 for B; these numbers are only workflow diagnostics. Checkpoint files contain optimizer state and remain outside Git under the existing repository ignore rules.

No GPU was available. CUDA end-to-end training and the OUYANG formal data-factory integration remain unverified. The WANG formal six-run ledger was left unchanged.
