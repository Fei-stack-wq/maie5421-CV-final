# OUYANG 数据模块接入与 GPU 联调

2026-11-10，在 WANG 分支合并 `ouyang/data-pipeline` 的 `f615077`，新增训练 factory 适配层。TONG、OUYANG 和 main 分支未改动。

## 接入方式

训练配置增加两个字段：

```yaml
dataset_factory: src.training.ouyang_factory:build_datasets
data_config: configs/ouyang_data_v1.json
```

适配层直接调用 OUYANG 的 `GenImageDataset` 和变换，按 TONG 的规则提供规范样本顺序及均衡 pilot 子集，并转发 `set_epoch`。训练仍由 TONG 的入口负责 shuffle；val 始终是 clean 单视图。完整配置见 [C/config.yaml](C/config.yaml)，可复用的入口配置见 [pilot_c_ouyang_cuda.yaml](../../examples/wang/pilot_c_ouyang_cuda.yaml)。

checkpoint 的 preprocessing 保存完整数据配置、配置指纹及文件哈希，注明 bicubic、EXIF 处理和 clean validation。ZHANG/ZHUO 加载这些 checkpoint 时应复用 OUYANG 的变换，不能套用旧 pilot 的 bilinear 预处理：

```python
from src.datasets import DataConfig, EvalTransform
from src.training.checkpoint import load_model

model, metadata = load_model(checkpoint_path, device="cuda")
config = DataConfig(**metadata["preprocessing"]["data_config"])
transform = EvalTransform(config, suite="clean")
```

## 本次实测

- Python 3.11.9、torch 2.3.0+cu121、Pillow 11.3.0、RTX 4060 Laptop 8GB。
- 合并测试100项：99通过，1项因本机有GPU而跳过无CUDA检查，包含OUYANG的50项测试和新增4项适配测试。
- 使用原清单的32张train、16张val；两个epoch共64组成对样本，B/C的ID、标签、image、view及实际变换参数逐项一致。16张验证输入均为clean。
- B、C(lambda=0)、C各2epoch，以及C的1+1epoch恢复全部通过；每轮2次实际更新，AMP跳步0。
- B与C(lambda=0)的模型、优化器、scaler、RNG、预测和history完全一致，耗时除外。C恢复与连续训练同样精确一致。
- 重载best checkpoint的概率误差为0；105个冻结tensor保持不变，17个可训练tensor已改变。

数值、源码哈希和测试计数见 [report.json](report.json)，逐epoch记录与预测在各子目录。验证源码提交为 `853be94`。这次使用正式数据模块，但仍是小样本工程联调，正式C模型为0/2，不以这里的AUC作效果结论。

复查命令如下，要求已配置CUDA训练环境、Pillow>=10.4和pytest，并准备对应图片与ImageNet权重缓存：

```powershell
python -m pytest -q
$env:TORCH_HOME = "$PWD/runs/torch-cache"
python -m tools.training.cuda_pilot --config examples/wang/pilot_c_ouyang_cuda.yaml --output-dir runs/wang_ouyang_repeat
```

脚本会生成仅含train/val的48行pilot清单；若图片未检出，可先加 `--prepare-only`，再用生成的 `sparse-paths.txt` 获取所选图片，具体方式同 [前次GPU联调](../wang-gpu-pilot/README.md)。输出目录不得含旧运行结果。

已收到OUYANG的全量QA和近重复候选处置记录，详见 [数据QA](../ouyang_qa.md)。本机仅核验本次48张图片；原始筛查和候选复核记录分别保留，不将它们改写成新的全量验收。清单在本机为CRLF，转换成LF后SHA256与OUYANG的记录一致；原清单未修改。

接下来共同确认两个种子、weight_decay、BN策略、AMP与实际更新预算，并准备本机完整Images/和GPU窗口，之后才启动六次正式训练。
