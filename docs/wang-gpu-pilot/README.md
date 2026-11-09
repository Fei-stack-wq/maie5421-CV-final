# C GPU 联调记录

2026-11-09，在 RTX 4060 Laptop 8GB 上完成。使用 TONG 的训练入口和临时 paired-view 模块，ImageNet 预训练 ResNet-18；32 张 train、16 张 clean-val，每轮 2 次更新，microbatch=4、累积4次。seed=5421 仅用于本次 pilot，正式共享种子尚未确认。

## 结果

- B、C（lambda=0）、C 各跑 2 epoch；C 另做 1 epoch 保存后恢复到第 2 epoch。
- B 与 C（lambda=0）的模型、优化器、scaler、随机状态、预测和逐轮记录完全一致，耗时除外。
- C 分段恢复与连续训练在以上字段上完全一致。
- 每次重载 best checkpoint 都复现保存的验证概率，最大误差为 0；105 个冻结 tensor 保持不变，17 个可训练 tensor 已更新。
- 最终运行每轮 2 次实际更新，AMP 跳步为 0。46 项单元测试中 45 项通过；本机有 CUDA，因此跳过“无 CUDA”环境检查。

| 运行 | 完整调用耗时 | PyTorch 峰值 allocated | PyTorch 峰值 reserved |
| --- | --- | --- | --- |
| B，2 epoch | 3.91 s | 249.6 MiB | 276.0 MiB |
| C（lambda=0），2 epoch | 3.26 s | 249.7 MiB | 276.0 MiB |
| C，2 epoch | 3.55 s | 249.7 MiB | 276.0 MiB |
| C，恢复前 1 epoch | 2.26 s | 249.6 MiB | 276.0 MiB |
| C，恢复后第 2 epoch | 2.40 s | 249.6 MiB | 276.0 MiB |

调用耗时包含数据检查和 checkpoint 写入；不含 Python 启动、首次下载及重载复核。显存是 PyTorch allocator 统计，不是整张显卡占用。小样本和缓存状态会影响时间，不能直接据此承诺完整数据的六次训练耗时。

第一次使用 AMP 默认初始 scale=65536，首轮两次更新均溢出跳过，恢复检查因没有参数变化而失败，记录见 [first-attempt.json](first-attempt.json)。随后在训练配置中增加 `amp_init_scale`，保持默认65536，pilot 显式设为1024后重跑通过。旧 checkpoint 缺少此字段时按原默认恢复，已加入回归测试。正式 B/C 应共同确认该设置并核对实际更新数与跳步数。

运行环境：Python 3.11.9、torch 2.3.0+cu121、torchvision 0.18.0+cu121、Pillow 10.2.0。最终测试源码提交为 `eb6bd2f`。各次配置、history、val predictions 和 summary 在对应子目录；源码和 checkpoint 哈希、精确比较结果见 [report.json](report.json)。权重不随 Git 上传。

## 复查

从仓库根目录执行，需要上述 CUDA 环境、所选图片和 torchvision 的 ImageNet 权重。第一次先生成48行清单及所需图片路径；有完整 Images/ 时不必稀疏获取。

```powershell
python -m tools.training.cuda_pilot --output-dir runs/wang_cuda_repeat --prepare-only
# 仅在使用稀疏检出时获取所选图片：
Get-Content runs/wang_cuda_repeat/sparse-paths.txt | git sparse-checkout add --no-cone --stdin
$env:TORCH_HOME = "$PWD/runs/torch-cache"
python -m tools.training.cuda_pilot --output-dir runs/wang_cuda_repeat
```

脚本拒绝覆盖已有运行结果。若权重不在缓存中，torchvision 会从 PyTorch 官方源下载。清单来自完整 manifest 的 train/val，原始 ID 与 SHA256 不变，48张文件已逐张核验；没有对测试集推理。这里的 AUC 只用于验证工作流，不能作为正式实验结论。

正式 C 模型仍为 0/2。后续要接入 OUYANG 的正式 factory、完成近重复检查并确认共同训练协议，再用完整训练集执行两个种子。
