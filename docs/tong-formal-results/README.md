# TONG：四组正式 A/B 训练与验收交接

**状态：4/4 已完成并独立验收。** 完成日期为 2026-11-18（北京时间）。
训练源码固定为 `d1a7752e557d77ba0f99a8d7243f4b51e166729b`，基于 WANG 集成提交
`0cb7e982c91e6be208dc3d18afc584988f5f0f5f`。本交接文档提交晚于训练，
不会改变模型中记录的训练源码版本。

## 结果与验收

每组使用完整 6,000 张 train 和 1,500 张 clean-val。最佳轮次在下表从 1 开始；
原始 checkpoint/summary 的 epoch 从 0 开始。

| 实验 | 完成轮数 | 最佳轮次（1 起） | 最佳 clean-val AUC | 实际更新次数 | AMP 跳步 |
| --- | --- | --- | --- | --- | --- |
| A_5421 | 7 | 5 | 0.9674808889 | 2625 | 0 |
| B_5421 | 7 | 5 | 0.9651288889 | 2625 | 0 |
| A_2026 | 8 | 7 | 0.9729191111 | 3000 | 0 |
| B_2026 | 4 | 2 | 0.9623946667 | 1500 | 0 |

A_5421、B_5421、B_2026 按 patience=2 提前停止，A_2026 达到 8 轮上限。
每个 best.pt 均在 CUDA 上重新加载，对完整 1,500 张 clean-val 独立重算；
保存预测与重载预测的最大概率差均为 **0.0**（容差 1e-5），所有验收通过。
last.pt 仅核对记录与哈希，没有额外做恢复训练实验。

这些是验证集选模型的结果，不是 held-out 测试或鲁棒性成绩。
原有 CPU/pilot 文档属于历史软件验证，不替代本目录中的正式实验结果。

## 环境、协议和计时

- GPU：NVIDIA A100-SXM4-80GB；C 参考使用 H800 80GB，硬件不同。
- Python 3.11.11、torch 2.3.0+cu121、torchvision 0.18.0+cu121、CUDA runtime 12.1；完整环境见 [environment.txt](environment.txt)。
- 官方 ImageNet ResNet-18；训练 layer4/head，冻结 BN running statistics。
- OUYANG v1；A 使用 clean-view BCE；B 使用两视图 BCE 均值；A/B consistency weight=0。
- AdamW lr=1e-4、weight_decay=0.01；microbatch 4 × accumulation 4，有效 batch 16。
- 最多 8 轮、clean-val AUC 选模、patience=2、AMP 初始 scale=1024、num_workers=0。
- 四组作业约 39.78 分钟，包含末次预检、训练、checkpoint I/O 和四次重载验收；不含环境准备、下载及先前 QA。
  各组 epoch train+val 时间见 [results.csv](results.csv)，不含 checkpoint I/O 和独立重载。
  已完成组的打包/传输与后续训练有重叠，耗时是本次执行记录，并非受控硬件基准。

完整 SHA256/解码/pHash 检查覆盖 12,100 张图像。原始 QA 保留 1 个已知近似候选；
未修改的 readiness gate 匹配已有的助手视觉误报复核记录后接受，没有绕过 gate，也没有新增人工签字。
见 [data-qa.json](data-qa.json) 和 [data-readiness.json](data-readiness.json)。
适用测试为 139 passed、1 skipped（CUDA 不可用场景）、1 deselected（已记录的 persistent-worker 问题，正式 num_workers=0）。

## 模型与文件交付

每组目录包含原始 config.yaml、history.csv、summary.json、val_predictions.csv 和独立 verification.json。
精确汇总见 [results.json](results.json)，逐个 checkpoint 的大小/哈希见 [checkpoint-manifest.json](checkpoint-manifest.json)，
四个完整模型 ZIP 的大小/哈希见 [artifact-archives.json](artifact-archives.json)。

完整模型归档保存在项目所有者的 DSW 工作区和 Windows D 盘，两份均已校验；当前没有 GitHub 模型下载地址。
请向项目所有者取得对应的 `TONG_<run_id>_artifacts.zip`，逐项核对本目录的 SHA256。
解压后模型位于 `models/<run_id>/best.pt`。所有 last.pt 与 best_epoch_NNN.pt 均已保留。
这些约 112 MB/份的模型按项目约定不进入 Git。
本次同步交接信息不等于已经授予其他成员 DSW/本机文件访问权限，也不冒称其他成员已收讫。

## ZHANG / ZHUO 加载接口

使用本分支源码与 requirements-training.txt 对应环境；将模型归档解压到当前项目根目录：

```python
from src.training.checkpoint import load_model
from src.datasets import DataConfig, EvalTransform

model, metadata = load_model("models/A_5421/best.pt", device="cpu")
transform = EvalTransform(
    DataConfig(**metadata["preprocessing"]["data_config"]), suite="clean"
)
```

模型输出 raw AI logit；sigmoid 后得到 P(AI)，阈值 0.5，real=0 / AI=1。
替换 run_id 即可加载其余模型。ZHANG 负责后续 seen/unseen/扰动评估及图表；
本目录 val_predictions.csv 仅用于重载核验。ZHUO 负责推理/CLI 集成。
本交接没有修改 C 模型或为其他成员声明任务完成。

报告可用的实际实验设置章节见 [tong-experimental-setup.md](../tong-experimental-setup.md)。
