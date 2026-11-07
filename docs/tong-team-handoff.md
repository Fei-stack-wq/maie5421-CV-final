# TONG：组员同步清单与运行边界

本文是需要共同确认的事项，不代表已经得到组员确认，也没有代替任何人完成正式实验。代码与验证入口见 `docs/tong-training.md`，本轮实测结果见 `docs/tong-extended-validation/`。

## WANG：先冻结共同训练协议

请确认以下取值，并由六次实验的统一配置与账本记录，而不是直接照搬 pilot：

- 两个共享 seed 是否使用 5421、2026；当前账本仍标为 proposed。
- AdamW lr=1e-4、weight_decay 是否为 0.01；有效 batch=16，GPU micro-batch 与 accumulation_steps 如何配对。
- 只训练 layer4 和 head。当前实现冻结**所有 BN running statistics**，但 layer4 BN 的 affine 参数可训练；是否采用这一共同策略。
- clean-val AUC 选择 checkpoint，patience=2、min_delta=0、最多 8 epoch；C 的 lambda 是否为 1。
- B/C 必须共用 sample 顺序、view 生成规则和最大更新预算；同时记录早停与 AMP skip，不能只比较配置中的 epoch 上限。
- `src.train` 已接入 WANG loss。额外 C pilot 仅验证接口，不替代 WANG 的两个正式模型，也不更新其正式账本为完成。
- 确认实际 GPU 型号、可用时段、六次运行顺序及失败重跑安排。CPU 的小样本时间不能直接作为 GPU 正式训练排期。

## OUYANG：接入正式数据与变换

需要可调用的 `dataset_factory: package.module:function`，详细契约见训练文档。重点确认：

- 接收完整 config，返回 train dataset、val dataset、可 JSON 序列化的 preprocessing/transform 元数据。
- `image_ids` 按对应 manifest split 的 image_id 排序，和训练入口一一对应；每项带 image_id、标量 label，建议使用 CPU float32 image/view `[3,224,224]`。
- train 提供 `set_epoch(epoch)`。同 seed、epoch、image_id 的 B/C view 完全相同，不能被 worker 数量或调用顺序改变。
- validation 的 image 必须为 clean 输入，且与推理使用同一预处理。
- 明确 JPEG 参数、降采样尺寸、操作概率/组合顺序与变换版本。现有 pilot 的 JPEG 50/70、112→224 resize 不是正式确认结果。
- 接收核验、感知近重复补查及数据版本冻结完成后，再设置 `protocol_confirmed: true`。精确 SHA256 去重不等于 pHash/DINO 近重复检查。

## ZHANG：对齐评估输入

- 统一 `real=0`、`AI=1`、阈值 0.5；`BinaryResNet18` 输出一个 raw AI logit。
- `load_model(best.pt)` 返回 eval 模型和 preprocessing metadata；预测前使用对应 metadata，不额外下载预训练权重。
- 正式预测 CSV 的 generator、transformation、seed、condition 等字段由评估端补齐。本分支的三列 val CSV 仅是 checkpoint 校验记录，不是最终交付 predictions.csv。
- 当前训练端 AUC 以 logits 排序，避免 float32 sigmoid 饱和形成伪并列。独立评估如只保存 p_ai，应确认 AUC 排序的一致性；可协商额外保留 logit 列。
- 仍需正式四个 A/B 模型，之后才做完整 seen/unseen/扰动结果和报告结论；本轮未对测试集进行模型评分。

## ZHUO：推理、恢复与模型交付

- 确认 Python/PyTorch/torchvision 环境和模型加载接口，先用临时 checkpoint 接入 CLI，再替换正式权重。
- 每次 run 使用空目录，避免覆盖已有结果；`best.pt` 可独立用于推理。
- 训练恢复需要同目录的 `last.pt` 与它引用的 `best_epoch_NNN.pt`，不能只复制 last.pt。只支持 epoch 边界恢复，且只允许扩展 epochs，不能更换 seed、预处理、manifest 或设备。
- 与组员确认权重的共享位置与访问权限。当前训练 checkpoint 含优化器，单份约 112 MB，按仓库现有规则不进入 Git。
- `requirements-tong.txt` 是本轮已测试版本；GPU 环境需安装适合驱动/运行时的官方 PyTorch wheel，再进行 CUDA 验证。

## FEI / 数据负责同学

- 本轮已重新运行原有完整数据校验器，报告另存，未覆盖 `Images/verification.json`；另核验 100 张 fixture，解码/哈希通过，与正式集的精确哈希交集为零。
- 上述检查只覆盖文件完整性、解码、尺寸/格式、数量、精确重复和 generator 划分，不代替内容层面的近重复或偏差对照。
- 当前检出的仓库包含图像文件，训练所需数据已经可读。若在其他电脑复现，仍应以实际文件和 manifest 核验结果为准。

## 当前环境能跑什么

已检查的是**当前云执行环境**，不是用户的个人电脑。实测为 9 个逻辑 CPU、约 9.7 GiB RAM、初始约 53.7 GiB 可用磁盘，PyTorch 为 CPU 版本，没有可用 CUDA。

可合理运行：自动化测试、实图小样本 A/B/C 训练、模型重载、epoch 恢复、单进程 DataLoader、一致性与冻结层检查、全图解码/哈希审计。

本环境不能验证：CUDA AMP、GPU 显存峰值、GPU 性能或 GPU 上的确定性。另外，实际尝试 num_workers=2 时，PyTorch 多进程的数据传输被本地 socket 权限阻断；该尝试已停止，num_workers=0 可以正常运行，不能宣称多 worker 检查通过。CPU 并非完全不能跑完整训练，但本轮不在尚未固定正式数据/协议的情况下长时间运行，并把它冒充最终四个模型。

用户个人电脑是否可用，仍需实际检查操作系统、内存、GPU 型号/显存、驱动和可用空间。不能由本次云环境结果推断用户电脑没有 GPU 或跑不了项目。
