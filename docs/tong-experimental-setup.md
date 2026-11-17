# Experimental setup: TONG chapter draft

**DRAFT: confirmed method, not completed A/B experiments.** The four A/B formal
runs remain NOT RUN. Remove this draft banner only after the actual-run evidence
below has been completed and reviewed. This chapter is intended for integration
into the team's six-page CVPR-format report, not a standalone final report.

## Confirmed design

We compare a clean-view baseline (A), paired-view augmentation (B), and a
paired-view consistency model (C). All conditions use an ImageNet-pretrained
ResNet-18 with a binary classification head. Only the final residual stage
(layer4) and classification head are optimized. Batch-normalization running
statistics remain frozen; affine parameters in layer4 remain trainable. The
positive label denotes AI-generated images, and accuracy uses a fixed 0.5
probability threshold.

The shared OUYANG v1 data pipeline uses the fixed manifest and split assignments.
Training contains 6,000 images and clean validation contains 1,500 images. The
full manifest has 12,000 images; the additional 100-image fixture is for software
validation and is not part of the training budget. The preprocessing, EXIF
handling, bicubic resizing and paired-view generation are those defined by
`configs/ouyang_data_v1.json` and its implementation, rather than the earlier
pilot transforms. Its configuration fingerprint is
`a5618523e8077bdb917d2ed63b483e20487014f2e5b59a2e7195f6d21622bbae`.
Held-out test data are not used for checkpoint selection.

A minimizes binary cross-entropy on the clean view. B minimizes the mean binary
cross-entropy across the paired views. The shared C reference adds a Bernoulli
Jensen–Shannon consistency term with weight 1; A and B use consistency weight 0.
This A/B implementation and configuration preserve the completed C protocol so
that the intended difference is the training condition rather than a new budget.

The planned common optimizer is AdamW with learning rate 1e-4 and weight decay
0.01. A microbatch of four and four accumulation steps give an effective batch
size of 16. Each condition has two runs, with seeds 5421 and 2026. The maximum
budget is eight epochs; checkpoint selection uses clean-validation AUC, with
patience two and minimum improvement zero. CUDA automatic mixed precision uses
initial loss scale 1024. We record actual optimizer updates, skipped AMP updates
and early stopping rather than assuming that an equal epoch cap implies an equal
number of completed updates. Each full epoch has 375 attempted updates.

## Actual-run fields: fill only from verified artifacts

| Run | State | Epochs / updates | Best epoch / val AUC | GPU / elapsed | Checkpoint SHA256 |
| --- | --- | --- | --- | --- | --- |
| A_5421 | NOT RUN | pending | pending | pending | pending |
| B_5421 | NOT RUN | pending | pending | pending | pending |
| A_2026 | NOT RUN | pending | pending | pending | pending |
| B_2026 | NOT RUN | pending | pending | pending | pending |

For each run, additionally record the exact source commit, manifest SHA256,
Python/PyTorch/torchvision/CUDA versions, AMP skips, stopping reason, all config
values, validation count, reload tolerance and observed maximum probability
error. Epoch indices in repository records are zero-based; state the convention
or convert consistently in the final paper. Elapsed time must name whether it is
training-only, includes reload, or includes queue/setup time.

Reference C runs were completed by WANG and are documented in
[the C result handoff](wang-formal-results/README.md). Their measurements must
not be copied into the empty A/B rows. Existing CPU/synthetic/pilot checks verify
software behavior only; they are not formal performance estimates.

## Remaining integration and review

1. Complete four authorized full-data A/B runs and independent reload checks.
2. Deliver verified checkpoints/configs/hashes to ZHANG and ZHUO; confirm the
   loading interface against the shared integrated branch.
3. Add actual hardware/runtime and final selected-epoch evidence to this chapter.
4. Cross-check the dataset description with OUYANG and loss terminology with WANG.
5. Incorporate ZHANG's held-out metrics/uncertainty/plots in the appropriate results
   chapter; do not infer them from validation AUC.
6. Review the combined report and prepare TONG's approximately 1.5-minute share of
   the team presentation. This draft does not claim the final report or rehearsal
   is complete.
