# 六次正式训练交接核对

WANG Yitian，2026-11-19。

A/B 四次与 C 两次正式训练均已交付，台账更新为 **6/6 completed**。本次核对配置、源码、训练日志、逐图验证记录和发布清单，没有重新训练或执行模型推理。

## 核对结果

- A/B 源码提交为 `d1a7752e557d77ba0f99a8d7243f4b51e166729b`，C 为 `b16b057ab82802da016639d605707fb482be72b1`。两版本的整个 `src/`、OUYANG 数据配置及 `requirements-training.txt` 无差异。
- 按相同种子比较完整运行配置，差异仅为实验条件和 JS 权重。共享种子为 5421/2026；预训练权重版本、冻结策略、AdamW、有效 batch 16、最多8轮、patience 2、AMP初始scale 1024等设置一致。
- 六次运行记录的 manifest SHA256 一致；训练集均6000张，验证集均1500张。六份逐图验证记录的 image ID、标签和行顺序完全一致，均为750张 real、750张 AI。
- A/B 的配置、summary、history、逐图验证 CSV 的原始字节 SHA256 均匹配 TONG 的逐次核验报告和结果清单；四个 best 模型的报告哈希匹配 GitHub Release 元数据。记录 ZIP（191312字节）的 SHA256 为 `1710be9f78c22b9fb6554e7f746cb2f873caac66a5a6aa05dcb1b67f6b4e0342`，与 GitHub 发布值一致。
- 六份历史日志均与 summary 中的轮数、最佳轮次和验证分数对应；AMP 跳步均为0。交付方的逐次模型重载报告均记录1500张验证图、最大概率误差0。本次复核了这些证据，没有重新下载或运行六个模型，也没有逐步重放初始化和训练视图。

| 运行 | 完成轮数 | 实际更新 | 最佳轮次（1起算） | 最佳 clean-val AUC |
| --- | ---: | ---: | ---: | ---: |
| A_5421 | 7 | 2625 | 5 | 0.9674808889 |
| B_5421 | 7 | 2625 | 5 | 0.9651288889 |
| C_5421 | 8 | 3000 | 6 | 0.9648533333 |
| A_2026 | 8 | 3000 | 7 | 0.9729191111 |
| B_2026 | 4 | 1500 | 2 | 0.9623946667 |
| C_2026 | 8 | 3000 | 6 | 0.9612248889 |

**实际更新预算并不相等。** A_5421、B_5421、B_2026因patience规则早停；两次C均达到8轮，同时满足早停条件。报告应写“共同最大预算与选模规则”，不能写“六次实际更新次数相同”。A/B使用A100-SXM4-80GB，C使用H800 80GB；各次耗时不能作为同硬件性能对照。A/B的逐轮计时之和不含checkpoint I/O与独立重载，台账没有将它填入C所用的整次运行耗时列；未提供的逐次起止时间留空。

这些 AUC 仅用于 clean-val 选模，不能据此得出 C 相对 B 的泛化或鲁棒性结论。数据 QA 原有的一对近重复候选及版本化处置记录继续保留，本次不新增人工签认，也不改变划分。

## 交接与下一步

- [六次台账](six-run-ledger.csv)、[机器可读核对记录](wang-six-run-review.json)。JSON含各次文件哈希和固定版本链接。
- [TONG A/B 模型与记录 Release](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/tag/tong-ab-formal-2026-11-18)、[固定版本交付说明](https://github.com/Fei-stack-wq/maie5421-CV-final/blob/d59c6e090456ae729fed2fae4d2d4dd36f21b2b2/docs/tong-formal-results/README.md)。记录包是发布前快照，当前下载位置以 Release 为准。
- [WANG C 模型与记录](wang-formal-results/README.md)。所有评估所需的 best 模型及原始训练记录均可通过 GitHub 获取，不要求共享训练服务器。

下一步由 ZHANG 对六个模型进行 held-out seen/unseen、扰动与偏差对照评估，并提供逐图预测、统计和图表。WANG收到后补写消融分析，保留上述预算及硬件差异。本次未合并 TONG 的 PR，也未接管其他成员的评估或推理工具任务。
