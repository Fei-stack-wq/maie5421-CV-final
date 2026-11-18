# ZMJ 正式 C 模型运行说明

更新于 2026-11-18。CLI 已接入 TONG 的 checkpoint 加载接口和 OUYANG 的统一预处理，在 CPU 上使用 WANG 的两份正式 C 模型进行单张 JPEG/PNG 图片预测。

本次 CLI 位于 `codex/zmj-local-smoke` 分支，基于 `wang/initial-preparation` 的 `0cb7e982c91e6be208dc3d18afc584988f5f0f5f` 开发。代码、运行说明和测试一同交付。

## 在当前电脑直接运行

打开终端，完整复制下面的命令。当前电脑已有 Python 环境、两份权重和示例图片。

```bash
cd /Users/zzzmj/Documents/ChatGPT/cv_final/maie5421-CV-final
.venv/bin/python -m cli.predict \
  --image Images/fixture/real/036_31cf018b81.JPEG \
  --checkpoint checkpoints/wang/C_5421_best.pt \
  --output outputs/formal_c/prediction_5421.json
```

本机实测输出如下。

```text
模型：C | seed=5421 | CPU | formal
图片：Images/fixture/real/036_31cf018b81.JPEG
预测：Real | P(AI)=0.000055
JSON：outputs/formal_c/prediction_5421.json
```

换成自己的图片时，修改 `--image` 后的路径。路径含空格时用双引号包住，例如 `--image "/Users/zzzmj/Desktop/my photo.jpg"`。输入图片可以放在仓库之外。

使用另一个种子的正式模型，运行以下命令。

```bash
.venv/bin/python -m cli.predict \
  --image Images/fixture/real/036_31cf018b81.JPEG \
  --checkpoint checkpoints/wang/C_2026_best.pt \
  --output outputs/formal_c/prediction_2026.json
```

`--output` 是可选项。提供时保存 JSON，同一路径已有结果会在本次预测成功后被替换。终端显示预测标签和 `P(AI)`。阈值固定为 0.5，`P(AI) >= 0.5` 输出 `AI-Generated`，其余输出 `Real`。

## 配置、预处理与结果追溯

正式 checkpoint 内含完整配置和预处理参数，CLI 自动读取。也可以附带该次训练的完整 YAML，检查它是否与权重一致。

```bash
.venv/bin/python -m cli.predict \
  --image Images/fixture/real/036_31cf018b81.JPEG \
  --checkpoint checkpoints/wang/C_5421_best.pt \
  --config docs/wang-formal-results/C_5421/config.yaml \
  --output outputs/formal_c/prediction_5421.json
```

使用 `C_2026_best.pt` 时，应配套 `docs/wang-formal-results/C_2026/config.yaml`。配置与权重不一致会报错。

推理直接复用 `src.training.checkpoint.load_model` 和 `src.datasets.EvalTransform`。预处理按模型中的参数执行 EXIF 方向校正、RGB 转换、短边缩放至 256、中心裁剪至 224×224、双三次插值和 ImageNet 归一化，使用 `clean` 模式。模型输出经 sigmoid 得到 `P(AI)`。运行设备固定为 CPU，适合当前 Mac 的本地演示。

JSON 保存预测、阈值、条件和种子、图片及权重的 SHA256、预处理参数、训练来源与实际推理设备。其中 `training_provenance` 的 CUDA 信息记录训练环境，顶层 `device` 记录本次 CPU 推理。

## 权重获取与换电脑运行

两份正式模型位于 `checkpoints/wang/`。验证本机权重完整性可以运行以下命令。

```bash
.venv/bin/python tools/download_c_models.py --verify-only
```

组员首次获取本次代码时，可以用以下命令下载源码和 100 张 fixture 示例。请在准备存放项目的目录运行。

```bash
git clone --branch codex/zmj-local-smoke --filter=blob:none --sparse \
  https://github.com/Fei-stack-wq/maie5421-CV-final.git
cd maie5421-CV-final
git sparse-checkout set cli src docs tests configs tools/training Images/fixture
```

使用 Python 3.12，在项目根目录创建环境、安装依赖和下载模型。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r cli/requirements-local.txt
.venv/bin/python tools/download_c_models.py
```

下载工具自动核对官方 SHA256。两份模型合计约 214 MiB，来源为小组的 GitHub Release。依赖安装、源码和权重准备完成后，单图预测可离线运行。当前 CLI 依赖文件记录 macOS arm64 / Python 3.12 环境，包含 PyTorch 2.15.0、torchvision 0.29.2 和 Pillow 13.0.0。[正式 C 模型发布页](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/tag/wang-c-formal-2026-11-12)

本机采用稀疏检出，已有 100 张 fixture 示例和本次核对的 4 张验证图。单图预测只需指定图片文件。完整数据集评估需要另行准备对应图片。

## 本地核对

运行 CLI 的两组测试。

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*cli.py' -v
```

测试覆盖两份正式权重、可选配置、重复运行、团队接口一致性、EXIF 方向和灰度图片、错误配置、输入文件保护，以及原有的临时模型流程。正式模型测试需要下载两份权重和 fixture 图片，缺少它们时会显示跳过。

2026-11-18 本机运行结果为 10 项通过，包括 4 项正式模型集成测试和 6 项原有流程测试。

本次通过 CLI 各运行两张真实照片和两张 AI 图片，共 8 次预测。参考值取自 WANG 的验证集记录。ZMJ 本机的核对结果保存在 `outputs/formal_c/validation_comparison.csv`，逐图 JSON 和 4 张原图也保存在本机的 `outputs/formal_c/`。这些是验证集样例核对，完整测试集、跨生成器和扰动结果由后续评估交付。

8 次分类结果均与发布记录一致，最大概率绝对差为 0.0009191036。本机使用 Mac CPU，发布记录使用 CUDA，环境版本也有差异。单图预测的 JSON 可以提供给 ZHANG 按图片和模型逐项对照。

## 常见情况

| 情况 | 处理方式 |
| --- | --- |
| 找不到 `cli` 或 `src` 模块 | 先进入项目根目录，再用 `.venv/bin/python -m cli.predict` 运行 |
| 找不到模型文件 | 检查路径，或运行 `tools/download_c_models.py` |
| 配置不匹配 | 使用同一种子的完整 `config.yaml`，也可省略 `--config` 自动读取模型配置 |
| 图片无法读取 | 检查文件是否完整，当前支持 JPEG 和 PNG |
| 输出路径覆盖输入 | 将 `--output` 改为 `outputs/` 下的独立 JSON 路径 |

## 保留的临时模型测试

原有流程测试仍可通过显式参数运行，用于开发阶段检查输入输出。

```bash
.venv/bin/python -m cli.predict \
  --image Images/fixture/real/036_31cf018b81.JPEG \
  --checkpoint outputs/local_smoke/best.pt \
  --config outputs/local_smoke/config.yaml \
  --allow-untrained \
  --output outputs/local_smoke/prediction.json
```

临时权重为随机初始化模型，输出会带有“仅流程测试”标记。正式演示使用前面的 `checkpoints/wang/C_*_best.pt` 命令。
