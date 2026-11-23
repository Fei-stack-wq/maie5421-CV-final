# Formal held-out evaluation

ZHANG Yiming evaluated the six published A/B/C checkpoints with seeds 5421 and 2026. The evaluator checks the released checkpoint and configuration hashes, verifies checkpoint metadata, and verifies the SHA-256 of every held-out image against the 12,000-image manifest. It reuses the shared ResNet-18 loader and OUYANG preprocessing. All six models see the same images in the same order. No checkpoint or threshold selection uses test data.

## Reproduce

Use Python 3.11 or newer with PyTorch, torchvision, NumPy, PyYAML, Pillow, pandas, SciPy, scikit-learn and Matplotlib. A CUDA-enabled build uses the GPU automatically. From the project root, with the image files in their original locations:

```sh
python -m cli.download_models
python -m eval.formal_predict --output eval/reproduced --batch-size 96 --threads 2
python -m eval.formal_metrics eval/reproduced
```

The evaluation branch includes the test images. The data tree matches commit `e7db18d0657865d5f344632534a3248cb49a6355`. For a checkout that contains only the manifest, restore the three test directories from the data commit (Git Bash):

```sh
git archive e7db18d0657865d5f344632534a3248cb49a6355 Images/test_seen Images/test_unseen_ADM Images/test_unseen_GLIDE | tar -xf -
```

## Evaluation protocol

Each split has 750 real and 750 generated examples. The seen split contains BigGAN and Stable Diffusion v1.4. The unseen splits contain ADM and GLIDE respectively. AI is positive and the hard decision is `p_ai >= 0.5`.

The 162,000 rows cover three splits, six models, and six preprocessing suites. Five suites are clean, JPEG q70, JPEG q50, half-resolution downsampling with bicubic reconstruction, and Gaussian blur with radius 1.0. The sixth suite is OUYANG's supplementary bias control, which resizes all inputs to 256 × 256 and re-encodes them at JPEG q95 before shared preprocessing. It reduces explicit size and encoding differences while also changing pixel content. Performance changes under this control are descriptive.

Metrics are accuracy, precision, recall, F1 and ROC-AUC. The CSV includes raw confusion counts per seed. Means average the two seed-specific metrics. The 95% percentile intervals use 2,000 label-stratified image bootstrap draws with random seed 5421. A draw uses the same IDs for all conditions, both seeds and every transformation within a split. The main contrasts are B−A, C−B, and each transformed suite minus its corresponding clean result. These are pointwise intervals for image sampling conditional on the six checkpoints and tested data distribution.

## Results files

- [Predictions](results/predictions.csv) include image ID, label, probability, generator, transformation, seed, condition, split and relative path.
- [Metrics CSV](results/metrics.csv) and [metrics JSON](results/metrics.json) include point estimates, confusion counts and confidence intervals.
- [Paired intervals](results/paired_intervals.csv) include comparison deltas and confidence intervals.
- [Clean ROC curves](results/clean_roc.png), [robustness curves](results/robustness_auc.png) and [paired AUC differences](results/paired_auc_differences.png) show the main comparisons.
- [Errors by source](results/clean_errors_by_source.csv) and [examples for review](results/clean_error_examples.csv) summarize clean-test mistakes.
- [Provenance](results/provenance.json) records model hashes, data and preprocessing fingerprints and inference runtime.
- [Results and analysis](RESULTS.md) contains the report text and presentation notes.
