# TONG: verified A/B experimental setup and results

Four authorized formal A/B runs were completed and independently verified on a single NVIDIA A100-SXM4-80GB GPU. The executed source commit was d1a7752e557d77ba0f99a8d7243f4b51e166729b. The reference C experiments used an NVIDIA H800 80GB; the hardware was different, while the corresponding protocol configurations were matched.

The environment was Ubuntu 22.04.5, Python 3.11.11, PyTorch 2.3.0+cu121, torchvision 0.18.0+cu121, CUDA runtime 12.1 and NVIDIA driver 570.133.20. Remaining direct dependencies match requirements-training.txt; the complete installed versions are included.

Each run used all 6,000 training images and 1,500 clean validation images. The 12,000-image formal manifest and additional 100-image fixture passed complete SHA256, decoding and pHash coverage checks. One known perceptual candidate remained recorded in raw QA; the unchanged readiness gate confirmed the versioned assistant visual false-positive review. This is not new human sign-off. Held-out test images were not used for model selection.

All runs used official ImageNet ResNet-18 IMAGENET1K_V1 weights, a single binary logit (AI positive), trainable layer4 and head, frozen BN running statistics, and the OUYANG v1 data pipeline. A optimized clean-view BCE; B averaged clean and transformed-view BCE. AdamW used lr=1e-4 and weight decay=0.01. Microbatch 4 with accumulation 4 gave effective batch 16, with up to eight epochs, clean-validation AUC selection, patience 2, deterministic execution, num_workers=0 and AMP initial scale 1024.

| Run | Completed epochs | Actual updates | Best epoch (1-based) | Best clean-val AUC | Recorded train+val minutes | Reload max probability error |
| --- | --- | --- | --- | --- | --- | --- |
| A_5421 | 7 | 2625 | 5 | 0.9674808889 | 8.20 | 0 |
| B_5421 | 7 | 2625 | 5 | 0.9651288889 | 10.51 | 0 |
| A_2026 | 8 | 3000 | 7 | 0.9729191111 | 9.40 | 0 |
| B_2026 | 4 | 1500 | 2 | 0.9623946667 | 6.08 | 0 |

All runs had zero skipped AMP updates. Each selected best.pt was independently reloaded on CUDA and evaluated on all 1,500 clean validation images. Predictions, labels, IDs, metrics, provenance, configurations and training records passed the unmodified verifier; the probability tolerance was 1e-5. last.pt was checked by records and hashes only, without an additional resume-training experiment. All versioned best checkpoints have been preserved.

The per-run minutes above sum recorded epoch training and validation timers; they exclude initial audit, checkpoint I/O and independent reload. The entire four-run job, including final preflight, I/O and reload checks, took 39.78 minutes. Setup and downloading are excluded. Packaging and network backup of completed groups overlapped later training, so these times describe this execution and are not controlled hardware benchmark measurements. These validation-selected results are not final held-out or robustness results, nor proof of identical performance across A100 and H800 hardware.

Exact values and provenance are in [formal results](tong-formal-results/README.md), including JSON/CSV, per-run verification reports and checkpoint SHA256. Subsequent held-out/robustness evaluation belongs to ZHANG; inference/CLI integration belongs to ZHUO.
