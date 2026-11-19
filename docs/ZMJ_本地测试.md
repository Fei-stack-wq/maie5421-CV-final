# ZMJ 六个正式 A/B/C 模型运行说明

更新于 2026-11-19。CLI 使用 TONG 的 checkpoint 加载接口和 OUYANG 的统一预处理，在 CPU 上运行 A/B/C × 两个种子的六个正式模型，支持单张 JPEG/PNG 图片预测。

本次 CLI 位于 `codex/zmj-local-smoke` 分支，基于 `wang/initial-preparation` 的 `0cb7e982c91e6be208dc3d18afc584988f5f0f5f` 开发。代码、运行说明和测试一同交付。

## 在当前电脑直接运行

打开终端，完整复制下面的命令。当前电脑已有 Python 环境、六份权重和示例图片。

```bash
cd /Users/zzzmj/Documents/ChatGPT/cv_final/maie5421-CV-final
.venv/bin/python -m cli.predict \
  --image Images/fixture/real/036_31cf018b81.JPEG \
  --checkpoint checkpoints/tong/A_5421_best.pt \
  --output outputs/six_models/example_A_5421.json
```

终端会显示模型条件、种子、预测标签和 `P(AI)`，同时保存 JSON。

换成自己的图片时，修改 `--image` 后的路径。路径含空格时用双引号包住，例如 `--image "/Users/zzzmj/Desktop/my photo.jpg"`。输入图片可以放在仓库之外。

例如，使用 C 条件的另一个种子，可以运行以下命令。

```bash
.venv/bin/python -m cli.predict \
  --image Images/fixture/real/036_31cf018b81.JPEG \
  --checkpoint checkpoints/wang/C_2026_best.pt \
  --output outputs/formal_c/prediction_2026.json
```

`--output` 是可选项。提供时保存 JSON，同一路径已有结果会在本次预测成功后被替换。终端显示预测标签和 `P(AI)`。阈值固定为 0.5，`P(AI) >= 0.5` 输出 `AI-Generated`，其余输出 `Real`。

六个模型使用相同命令入口，切换 `--checkpoint` 即可。

| 模型 | `--checkpoint` 路径 |
| --- | --- |
| A_5421 | `checkpoints/tong/A_5421_best.pt` |
| A_2026 | `checkpoints/tong/A_2026_best.pt` |
| B_5421 | `checkpoints/tong/B_5421_best.pt` |
| B_2026 | `checkpoints/tong/B_2026_best.pt` |
| C_5421 | `checkpoints/wang/C_5421_best.pt` |
| C_2026 | `checkpoints/wang/C_2026_best.pt` |

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

A/B 的配套配置通过下载命令一同准备，路径为 `checkpoints/tong/<模型名>/config.yaml`，例如 `checkpoints/tong/A_5421/config.yaml`。跨种子或跨 A/B/C 条件混用配置会报错。

推理直接复用 `src.training.checkpoint.load_model` 和 `src.datasets.EvalTransform`。预处理按模型中的参数执行 EXIF 方向校正、RGB 转换、短边缩放至 256、中心裁剪至 224×224、双三次插值和 ImageNet 归一化，使用 `clean` 模式。模型输出经 sigmoid 得到 `P(AI)`。运行设备固定为 CPU，适合当前 Mac 的本地演示。

JSON 保存预测、阈值、条件和种子、图片及权重的 SHA256、预处理参数、训练来源与实际推理设备。其中 `training_provenance` 的 CUDA 信息记录训练环境，顶层 `device` 记录本次 CPU 推理。

## 权重获取与换电脑运行

A/B 权重位于 `checkpoints/tong/`，C 权重位于 `checkpoints/wang/`。验证本机六份权重、六份配置和六份验证预测记录的完整性，可以运行以下命令。

```bash
.venv/bin/python -m cli.download_models --verify-only
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
.venv/bin/python -m cli.download_models
```

下载工具按 `cli/formal_models.json` 中固定的来源和 SHA256 核对全部文件。六份模型合计约 642 MiB，来源为小组的 GitHub Release。已存在且哈希一致的文件会复用，中断后可重新执行。配置及验证 CSV 从固定源码提交获取。依赖安装、源码和权重准备完成后，单图预测可离线运行。当前 CLI 依赖文件记录 macOS arm64 / Python 3.12 环境，包含 PyTorch 2.15.0、torchvision 0.29.2 和 Pillow 13.0.0。[正式 A/B 模型发布页](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/tag/tong-ab-formal-2026-11-18) · [正式 C 模型发布页](https://github.com/Fei-stack-wq/maie5421-CV-final/releases/tag/wang-c-formal-2026-11-12)

本机采用稀疏检出，已有 100 张 fixture 示例和本次核对的 4 张验证图。单图预测只需指定图片文件。完整数据集评估需要另行准备对应图片。

## 本地核对

运行 CLI 的两组测试。

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*cli.py' -v
```

测试覆盖六份正式权重、可选配置、重复运行、团队接口一致性、EXIF 方向和灰度图片、错误配置、输入文件保护，以及原有的临时模型流程。正式模型测试需要先运行模型下载命令并获取 fixture 图片，缺少权重或 fixture 时会显示跳过。

完整核对命令如下，会运行六个模型各 4 张固定验证图片，并复算每份已发布验证 CSV 的 AUC。

```bash
.venv/bin/python -m cli.verify_models
```

每个模型使用相同的两张真实照片和两张 AI 图片，共 24 次 CLI 预测，并与同机直接调用团队接口的结果逐项对照。首次执行会获取并核验这 4 张图片，后续复用本地文件。输出位于 `outputs/six_models/`，包括 `sample_predictions.csv`、`summary.json`、逐图 JSON 和样例图片。

每份验证 CSV 的 1,500 个图片 ID 和标签会与 manifest 核对，然后独立复算 AUC，与 checkpoint 的最佳验证 AUC 比较。本机重新执行模型的范围是 4 张样例；AUC 复算使用发布方保存的全部验证预测。报告记录 CPU 与发布环境之间的概率差异。完整测试集、跨生成器和扰动结果由后续评估交付。

2026-11-19 本机验收结果如下，10 项测试通过，全部 18 个发布文件的 SHA256 核对通过。24 次样例分类与发布记录一致，CLI 与同机团队接口的概率差为 0。相对发布记录的最大概率差为 0.0009191036。六份 CSV 共 9,000 条验证预测的 AUC 复算均与 checkpoint 记录一致。[完整核对报告](zmj-validation/summary.json) · [24 条逐图对照](zmj-validation/sample_predictions.csv)

| 模型 | 由验证 CSV 复算的 AUC | 样例分类与发布记录一致 |
| --- | --- | --- |
| A_5421 | 0.9674808889 | 4/4 |
| A_2026 | 0.9729191111 | 4/4 |
| B_5421 | 0.9651288889 | 4/4 |
| B_2026 | 0.9623946667 | 4/4 |
| C_5421 | 0.9648533333 | 4/4 |
| C_2026 | 0.9612248889 | 4/4 |

## 常见情况

| 情况 | 处理方式 |
| --- | --- |
| 找不到 `cli` 或 `src` 模块 | 先进入项目根目录，再用 `.venv/bin/python -m cli.predict` 运行 |
| 找不到模型或验证记录 | 检查路径，或运行 `python -m cli.download_models` |
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

临时权重为随机初始化模型，输出会带有“仅流程测试”标记。正式演示使用前面表格中的六份模型权重。
