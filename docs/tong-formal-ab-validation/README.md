# Formal A/B preparation validation (2026-11-17)

This is **software preparation evidence**, not a formal experiment result.
Formal A/B models completed: **0/4**. No GPU jobs were submitted.

- Base: WANG integration `0cb7e982c91e6be208dc3d18afc584988f5f0f5f`.
- CPU environment: Python 3.12.14, PyTorch 2.3.0+cpu, torchvision 0.18.0+cpu,
  NumPy 1.26.4, pytest 8.3.5; no CUDA device available.
- New runner/verifier tests: **39 passed**.
- Repository CPU suite: **138 passed, 2 skipped, 1 deselected**. The two skips
  require CUDA. Full-suite attempts stalled at
  `test_persistent_worker_observes_epoch_and_matches_main_process` and were
  interrupted; that one test was explicitly deselected in the completed run.
  This is a cloud-runtime limitation, not a passing result for that test and not
  a statement about the user's computer. Formal configs use `num_workers: 0`.
- Final recorded command: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
  -k 'not persistent_worker_observes_epoch_and_matches_main_process' --durations=10`.
- Python compilation, both CLI help pages, default read-only plan, shell syntax
  and Git whitespace checks passed. The plan created no run directory.
- Existing training/model/data implementation and OUYANG configuration are
  unchanged from the source used for completed C runs (`b16b057`).
- Independent review found and corrected zero-update false success, verifier
  output placement and the bare `runs/` output-root boundary. Any nonzero AMP
  skip count now requires budget review; `last.pt` is explicitly metadata/hash
  checked, not experimentally resume-validated by this new verifier.

The verifier tests use synthetic/mocked records and predictions. They do not
prove that four real 1,500-image reloads succeeded. No full dataset QA or real
A/B checkpoint reload was performed in this CPU preparation workspace.

Observed output is in [unit-tests.txt](unit-tests.txt). SHA256s for newly added
implementation/config/test/doc files are in [source-sha256.json](source-sha256.json).
