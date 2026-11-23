# TONG：ResNet18、A/B 训练与 checkpoint 接口

## 本次交付范围

本分支以 WANG 的 `53ad7b0` 为基线，保留其三次提交，未修改其 loss、实验账本或进度文件，也未修改主分支的数据集。新增内容限定为 TONG 的模型、训练、checkpoint、测试和联调文档。

- A：clean 图像的 BCEWithLogitsLoss。
- B：clean 与配对 view 的 BCE 平均值，调用 WANG 的 `paired_view_loss(..., consistency_weight=0)`。
- C：同一训练入口可调用 WANG 的一致性项；C 的实验配置与正式结果仍由 WANG 负责。
- ImageNet ResNet18，单 logit，0=real、1=AI；只训练 layer4 和二分类 head。
- 所有 BatchNorm 的 running statistics 固定，layer4 内 BN 的 affine 参数仍参与训练。此细节需纳入团队共同协议。
- AdamW、梯度累积、CUDA AMP、clean-val AUC 早停和逐 epoch 日志。
- 原子保存 state_dict checkpoint；按 epoch 边界恢复模型、优化器、AMP scaler、随机状态和早停状态。

代码使用了 AI 辅助编写和检查。自动化测试及小样本运行记录可复核；本说明不代表队友提交的作者身份判断。

## 安装与运行

从仓库根目录执行。已测试环境为 Python 3.12、CPU、torch 2.15.0+cpu、torchvision 0.29.2+cpu；依赖版本见 `requirements-tong.txt`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tong.txt
python -m unittest discover -s tests -v

python -m src.train --config examples/tong/pilot_a.yaml --output-dir runs/tong_pilot_A_5421
python -m src.train --config examples/tong/pilot_b.yaml --output-dir runs/tong_pilot_B_5421
```

首次训练会从 torchvision 官方来源下载 ImageNet 权重。可用 `TORCH_HOME` 指定缓存目录。CUDA 环境需使用 PyTorch 官方提供的匹配驱动/运行时 wheel；本次没有可用 GPU，未声称通过实际 CUDA 端到端训练。

两个示例都是 **32 张 train + 16 张 val、1 epoch 的流程 pilot**，不是正式四个 A/B 模型。种子 5421、weight_decay=0.01 和 JPEG/downsample 变换仅供当前 pilot；不意味着已冻结团队协议。不能把它们的 AUC 当成完整数据集的性能结论。

当前仓库实际包含 `Images/manifest.csv` 指向的 12,000 张图像。读取实际字段 `rel_path`，不是 README 的旧示例字段 `path`。启动时检查 train/val 类别、held-out generator、路径/ID 唯一性、内容泄漏，并重新核对 manifest 中声明的 SHA256。完整文件哈希审计不包含感知近重复检测，也不对测试集做模型推理。

## 模型与下游推理

```python
from src.models import BinaryResNet18
from src.training.checkpoint import load_model

model, metadata = load_model("runs/tong_pilot_A_5421/best.pt", device="cpu")
# images: float Tensor[N, 3, 224, 224]，按 metadata['preprocessing'] 预处理。
logits = model(images)          # [N] raw logits
p_ai = model.predict_proba(images)  # [N]，eval + no_grad
predictions = (p_ai >= metadata["threshold"]).long()
```

默认预处理：RGB，短边 resize 到 256（bilinear），中心裁剪 224，ImageNet mean/std。使用正式 factory 后，以 checkpoint 内记录的 preprocessing 元数据为准。`load_model` 不下载权重，使用 `torch.load(weights_only=True)`，校验 schema、architecture、label mapping 后严格加载 state_dict；不要改成反序列化整个模型对象。

固定阈值 0.5。AUC 以原始 logits 排序，避免 float32 sigmoid 饱和导致错误并列；导出的预测仍是 p_ai。所有早停和最佳模型选择只看 clean validation。

## 输出与恢复

每个独立 run 使用空目录，已有输出默认拒绝覆盖：

- `config.yaml`：完整配置。
- `history.csv`：每 epoch 的 loss、AUC、accuracy、实际更新数、AMP 跳过次数、耗时、早停状态。
- `best.pt`、`best_epoch_NNN.pt`：最佳模型及版本化恢复副本。
- `last.pt`：最近完成 epoch 的恢复点。
- `val_predictions.csv`：最佳 checkpoint 的逐图预测。
- `summary.json`：摘要、代码提交、工作区状态、库版本、manifest SHA256、数据审计和运行用途。

`last.pt` 是每个 epoch 的提交点。若在更新 best 与 last 之间中断，恢复时依据 last 指向的版本化 best 修复 best.pt 与预测文件。需要保留 `last.pt` 对应的 `best_epoch_NNN.pt`；不能只搬运 last.pt 用于恢复。独立推理只需 best.pt。

```bash
# 只允许增加原配置的 epochs，并仍使用同一输出目录。
python -m src.train --config path/to/extended.yaml \
  --output-dir runs/my_run --resume runs/my_run/last.pt
```

不支持 epoch 中途恢复。配置除 epochs 外、manifest、预处理和设备必须相同。已经触发早停的 run 不允许强行续训。相同环境中的断点恢复有回归测试；不保证不同 PyTorch 版本或 CPU/GPU 之间逐位一致。

Checkpoint 含优化器状态，文件较大，按仓库现有规则不进入 Git。本分支上传代码与小型验证记录，不上传模型权重或私聊材料。

## 与 OUYANG 数据模块对接

正式运行必须同时设置 `run_kind: formal`、`protocol_confirmed: true`、显式 `dataset_factory: package.module:function`，并将 train_limit/val_limit 设为 null。`protocol_confirmed` 是团队确认后的配置标记，不能用于跳过尚未完成的协议讨论。正式运行要求 ImageNet 预训练、有效 batch=16、最多 8 epoch，且 manifest 每一行都提供正确 SHA256。

Factory 接收完整 config 字典，返回 `(train_dataset, val_dataset, preprocessing_dict)`：

1. 两个 dataset 的 `image_ids` 与 manifest 对应 split 按 image_id 排序后的列表完全一致，且长度一致。pilot 如果使用 factory，则顺序须与 `balanced_subset` 的结果一致。
2. train 实现 `set_epoch(epoch)`；val 始终使用 clean 图像。每项返回 dict，含 `image`、标量 Tensor `label`、字符串 `image_id`。B/C train 还含 `view`。
3. image/view 为有限的浮点 Tensor `[3,224,224]`。包装器逐样本核对 ID、标签与形状；不会代替人工核实变换语义。
4. 对固定 seed、epoch、image_id，B/C 必须获得相同配对 view，不受全局随机状态、worker 数或调用顺序影响。不要让 A 少调用一个随机变换就改变后续 B/C 的 view。
5. preprocessing_dict 可 JSON 序列化，应记录真实预处理及变换参数/版本。任何变换变更都应改变此记录，恢复时会检查。

训练入口在 factory 创建后重新设置随机种子，保证 loader 初始化消耗随机数不会改变共享模型初始化；每 epoch 使用独立的 seed+epoch shuffle generator。B/C 使用同一清单顺序、配对规则与最大更新预算，实际早停/AMP 跳过次数仍需按日志核对。

仓库当前没有已接入此接口的 OUYANG 正式 factory，因此本实现故意将自带 PIL 变换标为 pilot，不擅自替代其正式模块。

## 自检与剩余工作

运行 `python -m unittest discover -s tests -v`：42 项测试，CPU 环境 40 项通过，WANG 的 2 项 CUDA 测试因无 GPU 跳过。覆盖冻结层/BN、A/B/C loss 接口、梯度累积尾批、AMP overflow 回退、AUC 并列/饱和、ID/标签/哈希校验、factory 顺序与随机性隔离、checkpoint 重载、早停和 epoch 恢复一致性。`compileall` 与 `git diff --check` 另行执行。

真实 ImageNet 权重与实际图像的 A/B pilot 结果、命令及重载验证见 `docs/tong-validation/`。这些记录仅证明当前 CPU 工作流连通，不证明正式实验精度。

正式 A/B 四次训练前仍需：

- 与 WANG 确认两颗共享种子、weight_decay、BN 策略及 B/C 共同预算；现有 ledger 的 5421/2026 仍标为 proposed。
- 与 OUYANG 接入并核对正式 paired-view factory、clean validation 和变换元数据。
- 在团队目标 GPU 上验证 CUDA AMP 与显存，冻结每个 run 的配置，再启动完整数据训练。
- 向 ZHANG/ZHUO 交接本页模型和 checkpoint 接口；正式训练完成后另行交付四份权重与逐 epoch 日志。

以上未完成事项没有被标成已完成，也没有修改 WANG 的正式六次实验账本。
