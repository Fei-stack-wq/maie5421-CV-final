# WANG Yitian 工作进度

更新：2026-11-09。项目第 1 天为 2026-11-05。

我负责一致性损失、C 配置和 C 的两个模型，以及六次训练的排期和记录。按《AI生成图像检测_五人两周工作安排》执行。

## 已完成

**11 月 5 日**：实现 Bernoulli JS 和双视图 BCE + JS，13 项损失测试通过。代码及用法见 [src/losses](../src/losses/)。

**11 月 9 日**：整合 TONG 的模型、训练和 checkpoint 代码，在 RTX 4060 Laptop 8GB 上完成 C 的真实图片 CUDA AMP 联调。

- 32 张训练图、16 张验证图，C 跑通 2 epoch，包括训练、保存、重载与预测核对。
- B 与 C（lambda=0）完全一致；C 断点续训与连续训练的模型、优化器、随机状态和预测完全一致。
- 冻结层及 BN 统计量保持不变，可训练参数实际发生更新。
- 发现默认 AMP 初始 scale 导致更新溢出跳过，增加 `amp_init_scale` 配置；pilot 使用1024后每轮2次更新、无跳步。旧 checkpoint 恢复兼容性已验证。
- 当前共46项单元测试，45通过；因为本机有GPU，跳过1项“无CUDA”检查。

配置见 [pilot_c_cuda.yaml](../examples/wang/pilot_c_cuda.yaml)，实测耗时、显存及复查步骤见 [GPU联调记录](wang-gpu-pilot/README.md)。这些小样本结果用于联调，正式C模型仍为 **0/2**。

## 接口与配置

TONG 的 `BinaryResNet18` 已输出单个 raw AI logit，训练入口已调用 `paired_view_loss`；`consistency_weight=0` 对应 B，`1` 对应 C。无需再手动转换输出。损失支持 `(N,)` 和 `(N,1)`，标签 `0=real, 1=AI`，两路样本顺序一致。

本次配置为有效 batch=16（microbatch=4，累积4次）、AdamW lr=1e-4、lambda=1。训练共用 TONG 的临时 JPEG/downsample 视图；它还不是 OUYANG 的正式变换。

## 下一步与交接

- **OUYANG**：接入正式 `dataset_factory`，对齐双视图返回、clean validation、变换参数及版本，并完成近重复检查。接口契约见 [训练说明](tong-training.md)。
- **TONG / WANG**：共同确认共享种子、weight_decay、BN冻结策略、AMP初始scale和更新预算，补齐正式A/B/C配置。GPU已联调通过，但正式设置尚未冻结。
- **训练排期**：确认GPU可用窗口、完整Images/可访问后安排六次运行；C的两次由我负责，A/B的四次由TONG负责。记录见 [six-run-ledger.csv](six-run-ledger.csv)，全部正式运行仍未开始；5421/2026仍为待确认种子。

正式模型只按 clean-val AUC 选择，推理阈值固定0.5。后续按分工交付两个C模型、六次运行记录，以及一致性方法与消融分析。
