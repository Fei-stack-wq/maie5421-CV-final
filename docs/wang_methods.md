# 一致性损失与训练方法（WANG Yitian）

更新：2026-11-12。

英文正文见 [wang_methods.tex](wang_methods.tex)，参考文献见 [wang_references.bib](wang_references.bib)。正文包含一致性目标、消融设计和训练设置，可放在 OUYANG 的成对视图说明之后；依赖 `amsmath`，将两条参考文献合入报告的 `.bib` 文件即可。若报告已引用 ResNet 或 JS 原论文，合并时统一引用键，避免重复条目。

## 损失和对照

每张图提供 clean 和 transformed 两个视图，标签均为 `0=real, 1=AI`。同一模型分别输出一个 raw AI logit，经 sigmoid 得到两个 AI 概率，再构成 `[P(real), P(AI)]` 的 Bernoulli 分布。

- **A**：只对 clean 视图计算 BCE。
- **B**：两路 BCE 的平均值。
- **C**：B 的损失加上两路分布的 JS 散度，`lambda=1`。

JS 使用自然对数，范围为 `[0, ln(2)]`；两路都反向传播，没有 teacher 或 detach。实现使用 logits 形式的 BCE，以及 `logsigmoid` / `logaddexp`；半精度输入在损失计算前转为 FP32。损失按原始图片数取平均，不能将两路 BCE 相加后再加同样权重的 JS，否则会改变两项的相对权重。

B/C 应共享初始化、样本顺序和逐图变换。A/B 对照同时改变监督视图和每张图的计算量，不能称为等计算量实验。相同的 epoch 上限也不能直接证明实际更新次数相同，需要对照各次运行日志。

## 已执行的 C 设置

| 项目 | 实际设置 |
| --- | --- |
| 种子 | 5421、2026 |
| 模型 | ImageNet ResNet-18，单 logit 输出 |
| 冻结 | 仅 layer4/head 可训练；全部 BN running statistics 固定，layer4 BN affine 可训练 |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01 |
| 批次 | microbatch=4，累积4次，有效 batch=16 张原图 |
| 预算与选择 | 最多8轮，clean-val AUC选模型，patience=2，min_delta=0 |
| 精度与设备 | FP16 AMP，初始scale=1024，单张H800 80GB |
| 实际执行 | 两个种子均8轮、3000次更新、AMP跳步0；最佳模型均为第6轮 |
| 推理 | 单视图，sigmoid后阈值0.5 |

共同种子、模型和优化器等属于已确认的 A/B/C 协议；以上“实际执行”仅指本人完成的 C 实验，不代表已核验 TONG 的四次 A/B 正式运行。A/B 交付后还需核对 AMP 初始 scale、实际更新及早停记录。

## 依据与后续

损失对应 [consistency.py](../src/losses/consistency.py)，模型冻结对应 [resnet18.py](../src/models/resnet18.py)，正式配置见 [C_5421](../configs/wang/c_5421.yaml) 和 [C_2026](../configs/wang/c_2026.yaml)。配置、逐轮日志及独立重载核验见 [正式结果目录](wang-formal-results/README.md)；权重通过该目录链接的 GitHub Release 获取，无需访问训练服务器。

JS 定义参考 [Lin (1991)](https://doi.org/10.1109/18.61115)，骨干模型参考 [He et al. (2016)](https://arxiv.org/abs/1512.03385)。本文贡献是本项目的监督双视图训练实现和对照设计，不将 JS 或 ResNet 作为新提出的方法。

方法部分已经可以合入报告。当前 clean-val AUC 仅用于选模型；C 相比 B 的收益、未见生成器泛化及扰动鲁棒性结论，要等 A/B 模型与 ZHANG 的正式测试结果后再写。
