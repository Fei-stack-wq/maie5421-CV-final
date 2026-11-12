# WANG Yitian 工作进度

更新：2026-11-12。项目第 1 天为 2026-11-05。

我负责一致性损失、C 配置和 C 的两个模型，以及六次训练的排期和记录。按《AI生成图像检测_五人两周工作安排》执行。

## 已完成

**11 月 5 日**：实现 Bernoulli JS 和双视图 BCE + JS，13 项损失测试通过。代码及用法见 [src/losses](../src/losses/)。

**11 月 9 日**：整合 TONG 的模型、训练和 checkpoint 代码，在 RTX 4060 Laptop 8GB 上完成 C 的真实图片 CUDA AMP 联调。

- 32 张训练图、16 张验证图，C 跑通 2 epoch，包括训练、保存、重载与预测核对。
- B 与 C（lambda=0）完全一致；C 断点续训与连续训练的模型、优化器、随机状态和预测完全一致。
- 冻结层及 BN 统计量保持不变，可训练参数实际发生更新。
- 发现默认 AMP 初始 scale 导致更新溢出跳过，增加 `amp_init_scale` 配置；pilot 使用1024后每轮2次更新、无跳步。旧 checkpoint 恢复兼容性已验证。
- 当前共46项单元测试，45通过；因为本机有GPU，跳过1项“无CUDA”检查。

**11 月 10 日**：在本分支整合 OUYANG 的数据模块，新增 factory 适配，接入其固定预处理与成对视图。100项合并测试中99项通过，1项因本机有GPU而跳过。使用实际数据模块再次完成GPU联调：64组成对样本的B/C输入逐项一致，C的续训和重载检查通过，每轮2次更新且无AMP跳步。

**11 月 12 日**：在Superpod H800完成两个正式C实验，均训练8轮、3000次更新、AMP跳步0，最佳模型均来自第6轮。种子5421/2026的最佳clean-val AUC分别为0.9648533333和0.9612248889；独立重载全部验证预测，最大误差均为0。正式C模型已完成 **2/2**，配置、日志和权重位置见 [正式结果与交接](wang-formal-results/README.md)。

历史接入方式及证据保留在 [数据模块联调记录](wang-ouyang-integration/README.md) 和 [前次GPU联调](wang-gpu-pilot/README.md)。这些pilot不计入正式模型数量。

## 接口与配置

TONG 的 `BinaryResNet18` 已输出单个 raw AI logit，训练入口已调用 `paired_view_loss`；`consistency_weight=0` 对应 B，`1` 对应 C。无需再手动转换输出。损失支持 `(N,)` 和 `(N,1)`，标签 `0=real, 1=AI`，两路样本顺序一致。

本次配置为有效 batch=16（microbatch=4，累积4次）、AdamW lr=1e-4、lambda=1。现已使用 OUYANG v1 的 bicubic/EXIF 预处理及成对视图，配置指纹随checkpoint保存；不再使用临时变换。

## 下一步与交接

- **数据接入已完成**：OUYANG模块与训练入口已联调；全量QA和候选处置记录已收到。本机完整Images/尚未检出，正式训练前仍需本机接收核验。
- **参数确认**：11月11日已同意TONG交接清单中的参数，种子确定为5421/2026；C配置已生成，见 [正式训练设置](wang-training-protocol.md)。AMP初始scale=1024是本分支的GPU实测设置，A/B执行时需显式对齐。
- **训练位置**：11月12日确定在HKUST Superpod执行本人两次C实验，使用Slurm申请1张GPU，两个种子顺序执行。先完成服务器数据和环境核验，再启动正式训练；A/B四次由TONG负责。
- **运行记录**：见 [six-run-ledger.csv](six-run-ledger.csv)，本人两次C已完成并通过独立核验；A/B的状态和交付由TONG更新。服务器作业入口及依赖关系见 [运行说明](../scripts/superpod/README.md)。

正式模型只按 clean-val AUC 选择，推理阈值固定0.5。下一步向ZHANG/ZHUO交接C权重，跟踪A/B交付；方法章节可以继续整理，消融结论等待正式评估。
