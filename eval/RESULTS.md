# Results and analysis

## Evaluation protocol

We evaluated six fixed ResNet-18 checkpoints on three byte-deduplicated GenImage test splits of 1,500 images each. The seen split contains BigGAN and Stable Diffusion v1.4, and the unseen splits contain ADM and GLIDE. We compared clean inputs with four image perturbations and a supplementary format and size control. Within each split, every model and input condition used the same image IDs. AI is the positive class and the decision threshold is 0.5. Point estimates average the scores of seeds 5421 and 2026. Paired 95% percentile intervals resample images within label strata, with the same draws across models and both seeds.

## Clean cross-generator performance

| Test split | A AUC | B AUC | C AUC | A F1 | B F1 | C F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen | 0.967 | 0.957 | 0.954 | 0.904 | 0.890 | 0.890 |
| Unseen ADM | 0.576 | 0.528 | 0.558 | 0.340 | 0.228 | 0.298 |
| Unseen GLIDE | 0.960 | 0.878 | 0.878 | 0.894 | 0.718 | 0.773 |

A had the highest clean ROC-AUC on all three splits. For A, the difference between clean GLIDE and ADM was 38.5 percentage points. On clean ADM, recall at 0.5 ranged from 14.1% to 22.7% across conditions. C improved ADM AUC over B by 3.06 points (95% CI 2.02 to 4.15), with a mean AUC of 0.558. On clean GLIDE, C and B had similar AUC (difference 0.03 points, CI −0.67 to 0.71), while recall increased from 62.3% to 73.0% and the real-image false positive rate increased from 11.2% to 15.7%. [Metrics](results/metrics.csv) · [Paired intervals](results/paired_intervals.csv)

## Robustness and ablations

On the seen split, B increased AUC over A by 2.25 points for JPEG q50 (CI 1.52 to 3.08), 6.41 points for half-resolution downsampling (CI 5.26 to 7.62), and 6.39 points for blur (CI 5.24 to 7.64). C increased AUC over B under downsampling and blur in all three splits. The increases ranged from 1.15 to 2.84 points, and all six pointwise intervals were above zero. C−B was close to zero for either JPEG quality on seen and GLIDE, with intervals spanning zero. The benefit of consistency depends on the generator and perturbation.

At the fixed threshold, A classified 90.0% of the seen real photos as AI after blur. B and C reduced that false positive rate to 53.6% and 57.2%, respectively. AUC improved from B to C on this suite while accuracy fell from 71.3% to 70.0%. Thus ranking and fixed-threshold decisions give different pictures of robustness. [Robustness curves](results/robustness_auc.png) · [Paired AUC plot](results/paired_auc_differences.png)

## Bias control and representative errors

With all images resized to 256 × 256 and re-encoded at JPEG q95, A retained AUCs of 0.960 on seen and 0.949 on GLIDE, and reached 0.586 on ADM. The ADM shortfall persisted under this control. On the clean seen split, A missed 0.9% of BigGAN samples and 14.4% of SD v1.4 samples, averaged over the two checkpoints. The source-level error rates and high-confidence mistakes provide image IDs for independent review. [Errors by source](results/clean_errors_by_source.csv) · [Representative errors](results/clean_error_examples.csv)

## Interpretation limits

The test splits contain different real images as well as different fake sources, so gaps between splits reflect distribution shift across both sides of the task. The bias control combines resizing and re-encoding. A performance change under that control can reflect changes in semantic details, generator artifacts and low-level source cues. The source records document byte-level verification and mark perceptual deduplication as incomplete. The ablation compares the released checkpoints under shared nominal settings. The releases used different training code revisions and hardware, and early stopping produced different realized update counts. The confidence intervals are pointwise and describe image sampling for the two fixed seeds.

The published predictions use an RTX 4070 SUPER and CUDA. On the clean seen split, an independent CPU run in a second local environment agreed on 8,999 of 9,000 thresholded outputs. The largest absolute probability difference was 0.00617. See the [inference provenance](results/provenance.json), [A/B handoff](https://github.com/Fei-stack-wq/maie5421-CV-final/pull/3) and [C handoff](../docs/wang-formal-results/README.md).

## Presentation notes

我们用同一份测试清单评估了六个正式模型。每个测试集 1,500 张图，原图和扰动图使用相同图片 ID 做配对比较。先看原图，A 在 seen 和 GLIDE 的 AUC 都约为 0.96，但在 ADM 只有 0.576，说明对不同留出生成器的泛化差异很大。再看扰动，在 seen 测试上，普通增广让重度 JPEG、降采样和模糊下的 AUC 有所提高。一致性项在三个测试集的降采样和模糊上均进一步提高 AUC，但 JPEG 收益因生成器而异。最后，固定阈值下的错误也值得看。seen 的模糊图上，A 把约九成真实照片判成了 AI，B 和 C 降低了这个比例，但误报仍高。我们的区间衡量这批图片上的抽样变化，报告也同时列出了两个种子的指标。
