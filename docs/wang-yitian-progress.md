# WANG Yitian 工作进度

2026-11-05 · 第 1 天

我负责一致性损失、C 配置和 C 的两个模型，以及六次训练的排期和运行记录。时间安排按《AI生成图像检测_五人两周工作安排》执行。

## 今天完成

一致性损失已实现，代码见 [src/losses](../src/losses/)，提交 [16be274](https://github.com/Fei-stack-wq/maie5421-CV-final/commit/16be27498bd7beb2653528048ce40dc727c38c69)。

- `bernoulli_js_divergence`：计算原图与变换图预测之间的 Bernoulli JS 散度，两路都保留梯度。
- `paired_view_loss`：两路 BCE 取平均，再加一致性项；`consistency_weight=0` 对应 B，`1` 对应 C。
- 13 项测试通过，包括公式对照、零散度、对称性、梯度检查、极端数值和 CUDA 混合精度。测试使用模拟输入，尚未进行真实数据联调或正式训练。

复查命令：

```bash
python -m unittest discover -s tests -p test_consistency.py -v
```

六次训练的记录表已建好，见 [six-run-ledger.csv](six-run-ledger.csv)。目前均未开始；表中的 5421、2026 和执行顺序尚未共同确认。

## 接入方式

损失函数接收原图和变换图的原始 AI logit，形状为 `(N,)` 或 `(N, 1)`；标签为 `0=real, 1=AI`，两路样本顺序要一致。

```python
from src.losses import paired_view_loss

loss = paired_view_loss(logits, view_logits, labels, consistency_weight=1.0)
loss.backward()
```

如果模型输出 `[real, AI]` 两个 logit，先取 `output[:, 1] - output[:, 0]`。公式和完整接口说明见 [损失模块说明](../src/losses/README.md)。

## 接下来需要对齐

- **TONG**：确认模型输出和训练入口的配置字段，把损失接入训练；对齐有效 batch、梯度累积、checkpoint 和随机状态恢复方式。
- **OUYANG**：确认 loader 的双视图与标签返回方式，以及变换参数和随机性控制，让 B/C 使用相同批次与视图。
- **训练安排**：共同确定两个种子和 GPU 可用时间。第 3 天试跑后按实测耗时排六次训练；C 的两次由我负责，A/B 的四次由 TONG 负责。

下一步完成 A/B/C 配置和 C 联调。按分工，第 3 天通过完整试跑，第 7 天前交付两个 C 模型和六次运行记录，第 8–12 天完成一致性方法与消融分析。正式 checkpoint 只按验证集 AUC 选择，推理阈值固定 0.5。
