# 欧阳迪个人展示内容

作者：OUYANG Di 欧阳迪。用于全组约 10 分钟展示中的个人约 1.5 分钟部分。内容覆盖数据与扰动方法，衔接 FEI 的数据来源介绍和 TONG/WANG 的模型与训练介绍；模型结果由 ZHANG 的正式评估提供。

## 单页展示建议

标题：Data preprocessing and paired views

- Manifest → EXIF/RGB → resize short side 256 → centre crop 224 → ImageNet normalisation.
- B/C receive the same clean image and transformed view: JPEG, downsampling, or both.
- Shared seed + epoch + image ID reproduce B/C views; batch and update budgets are controlled separately.
- Fixed tests: clean, JPEG q70/q50, 0.5× downsampling, Gaussian blur radius 1.0.
- Extra bias check: resize 256×256 + JPEG q95; reception QA and pHash candidates are recorded separately.

配图建议：选同一张 fixture 图片，展示 clean、JPEG q50、0.5× 降采样、blur 的四列样例。标明这是输入示例，不是模型预测结果。旁边用两条箭头表示同一 crop 分别进入 clean 分支和变换分支，再交给 B/C。

## 中文讲稿

我负责把已有数据接入训练和评估，并保证各个实验使用一致的输入规则。

首先，我们按 manifest 中的划分、路径和标签读取图片，保留图片 ID 供后续配对分析。所有条件都先校正方向、转成 RGB，把短边缩放到 256，中心裁剪到 224，再做 ImageNet 归一化。

对于 B 和 C，同一张图片返回干净视图和变换视图。变换等概率选择 JPEG、降采样或两者组合。随机参数由共享种子、epoch 和图片 ID 确定，不依赖标签、实验条件或读取顺序，所以 B、C 能使用相同视图；训练方还需要对齐批次和更新预算。

测试固定使用 clean、JPEG q70 和 q50、二分之一降采样，以及半径为 1 的高斯模糊。我们还提供统一尺寸并重编码为 JPEG q95 的偏差对照，因为真实图与生成图的格式和原生尺寸不同。这个对照只能缓解偏置，不能保证去除已有痕迹。

最后，数据 QA 分别记录清单、文件哈希与解码，以及跨划分的 pHash 相似候选。候选需要人工复核。我的交付让下游可以复用同一套处理；是否提高泛化和鲁棒性，需要由正式模型结果回答。

## English script

My part connects the prepared dataset to training and evaluation, using
consistent input rules across experiments.

We read the split, path, and label from the manifest and retain image IDs for
paired analysis. Every condition uses RGB conversion, resizing the shorter
side to 256, a 224-pixel centre crop, and ImageNet normalisation.

For B and C, each image returns a clean view and a transformed view. We select
JPEG, downsampling, or both with equal probability. A shared seed, epoch, and
image ID determine the parameters independently of the label, condition, or
access order. B and C can therefore receive identical views, while the
training code separately matches batches and update budgets.

The fixed tests are clean, JPEG quality 70 and 50, half-scale downsampling,
and Gaussian blur with radius one. We also provide a uniform-size,
JPEG-quality-95 control because real and generated images differ in format
and native resolution. This reduces confounds but cannot erase their earlier
compression or resolution history.

Finally, reception QA separates manifest checks, file hashes and decoding,
and perceptual-hash candidates across splits. Similar candidates require
manual review. This delivers a reusable, controlled data pipeline; whether it
improves generalisation or robustness remains a question for the final model
results.

## 展示前核对

请按实际 QA JSON 和联调记录介绍完成状态，不把原仓库的旧核验记录当作本次结果。只有全量图片检查和候选复核完成后，才能口头报告相应结论。展示样例应使用同一张图片及固定参数，并与评估采用相同预处理。中英文讲稿择一使用，不在同一 90 秒段落中连续朗读两版。
