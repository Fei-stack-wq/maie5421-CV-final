# 两个正式 C 模型交付

两个正式C模型、配置、逐轮日志、逐图验证预测和QA记录均通过本仓库交付，不需要访问训练服务器。

## 下载模型

[GitHub Release](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/tag/wang-c-formal-2026-11-12)

- [C_5421_best.pt](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/download/wang-c-formal-2026-11-12/C_5421_best.pt)
- [C_2026_best.pt](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/download/wang-c-formal-2026-11-12/C_2026_best.pt)
- [配置和记录打包下载](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/download/wang-c-formal-2026-11-12/wang_c_records.zip)

也可在仓库根目录运行以下命令，自动下载并检查SHA256：

```bash
python tools/download_c_models.py
```

默认保存到checkpoints/wang。文件哈希见 [checkpoint-manifest.json](checkpoint-manifest.json)。模型较大，作为Release附件交付；源码分支为wang/initial-preparation，训练源码固定为b16b057。

## 运行记录

2026-11-12，在Superpod NVIDIA H800 80GB上训练。每次使用6000张train和1500张clean-val。

| 种子 | 完成epoch | 最佳轮次 | 最佳clean-val AUC | 实际更新 | AMP跳步 |
| --- | --- | --- | --- | --- | --- |
| 5421 | 8 | 第6轮 | 0.9648533333 | 3000 | 0 |
| 2026 | 8 | 第6轮 | 0.9612248889 | 3000 | 0 |

checkpoint中epoch从0计，最佳epoch均为5。两个模型均重载复核1500条验证预测，最大误差0。训练作业耗时19分01秒，独立核验45秒。以上是验证集选模型的指标，不是测试集或鲁棒性结论。

每个种子目录包含config.yaml、history.csv、summary.json和val_predictions.csv；本目录还包含独立核验报告、依赖版本、测试计数、数据接收及QA记录。

## ZHANG / ZHUO 接入

```python
from src.training.checkpoint import load_model
from src.datasets import DataConfig, EvalTransform

model, metadata = load_model("checkpoints/wang/C_5421_best.pt", device="cpu")
transform = EvalTransform(DataConfig(**metadata["preprocessing"]["data_config"]), suite="clean")
```

模型输出一个raw AI logit；对其sigmoid得到P(AI)，阈值固定0.5，AI为正类。替换文件名即可加载另一个种子。评估时复用OUYANG变换和稳定图片ID；当前val_predictions.csv是验证集记录，正式测试预测由ZHANG生成。依赖见requirements-training.txt。
