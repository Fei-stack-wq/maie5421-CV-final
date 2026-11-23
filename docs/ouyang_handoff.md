# 欧阳迪数据模块交接说明

作者：OUYANG Di 欧阳迪。课程：MAIE5421 Computer Vision Project 6。交付范围：按清单读取图片、统一预处理、B/C 成对视图、测试扰动、数据接收 QA、报告方法内容和个人展示稿。

本模块将 FEI 提供的数据接入 TONG/WANG 的训练和 ZHANG/ZHUO 的评估与推理。它保留原始图片、清单和划分，通过同一套像素处理规则控制 A/B/C 的输入差异。模型训练、checkpoint、预测 CSV 和最终指标由相应成员提供。

本次验收：50 项测试通过；正式集及 fixture 共 12,100 张全部完成文件核验；pHash 唯一候选经助手视觉复核判为不同内容。详细结果、限制与复核图见 [`ouyang_qa.md`](ouyang_qa.md)。

## 交付文件

| 文件或目录 | 用途 | 接收方 |
|---|---|---|
| `src/datasets/` | 数据集、预处理、成对视图和固定测试扰动 | TONG、WANG、ZHANG、ZHUO |
| `configs/ouyang_data_v1.json` | 固定数据与变换参数 | 全组 |
| `tools/dataset/qa_received.py` | 清单核验、接收核验和感知近重复候选检索 | FEI、TONG、WANG |
| `docs/ouyang_methods.tex` | 可并入六页 CVPR 报告的英文方法内容 | FEI |
| `docs/ouyang_presentation.md` | 单页展示要点与约 90 秒中英文讲稿 | ZHUO |
| `docs/ouyang_qa.md`、`artifacts/ouyang/` | 本次实测结果、环境版本和候选视觉复核证据 | 全组 |
| `tests/test_datasets.py` | 数据接口及可复现性验证 | 下游接入人员 |

## 运行与接口试跑

以下命令在仓库根目录运行，`python` 应指向已安装本模块依赖的环境。

```powershell
python -m pip install -r requirements-data.txt
python -m pytest -q
python -m tools.dataset.smoke_data --manifest Images/fixture/manifest_fixture.csv --root Images --split fixture --config configs/ouyang_data_v1.json --output-dir artifacts/ouyang/smoke
python tools/dataset/qa_received.py --metadata-only --output artifacts/ouyang/qa_metadata.json
python tools/dataset/qa_received.py --check-images --phash --output artifacts/ouyang/qa_received.json
```

试跑会写出 `smoke.json` 和 `preview.png`，用于核对真实 fixture 的输出形状、成对视图及固定扰动。该试跑验证数据模块，不代表训练、checkpoint 恢复或指标链路已通过。Windows 上使用多个 DataLoader worker 时，将创建 loader 和遍历的入口置于 `if __name__ == "__main__":` 中；每个 epoch 完整消费或结束旧迭代器后，再设置下一个 epoch。

缺图时运行 `python -m tools.dataset.receive_images`，可从固定提交 `e7db18d0657865d5f344632534a3248cb49a6355` 接收两份清单对应的图片，逐张核对长度和 SHA256。已存在且哈希一致的文件会跳过，已存在但内容不符的文件会报错并保留，支持重复运行。首次完整接收约 1.95 GB。只需开发样例时添加 `--fixture-only`。旧 README 和 dataset card 的“图片不入 Git”说明与当前提交的文件列表不符。

## 读取规则与返回结构

使用 `Images/manifest.csv` 的 `split`、`rel_path` 和 `label`；`rel_path` 始终相对于 `Images/`，不通过文件夹名称推断标签。`label=0` 表示 Real，`label=1` 表示 AI。图片 ID、生成器与划分随样本返回，方便预测对齐。

正式划分为 `train`、`val`、`test_seen`、`test_unseen_ADM`、`test_unseen_GLIDE`。fixture 单独使用 `Images/fixture/manifest_fixture.csv`；其路径仍相对于 `Images/`，因此必须显式设置 `root="Images"`。fixture 用于开发，不加入正式训练或最终评估。

```python
from src.datasets import DataConfig, GenImageDataset, make_dataloader

config = DataConfig.from_json("configs/ouyang_data_v1.json")

train = GenImageDataset(
    "Images/manifest.csv", split="train", condition="B", seed=5421,
    config=config
)
loader = make_dataloader(train, batch_size=16, num_workers=0)

for epoch in range(8):
    train.set_epoch(epoch)
    for batch in loader:
        images = batch["image"]       # float32 [N, 3, 224, 224]
        views = batch["view"]         # 同一图片的变换视图，B/C 训练使用
        labels = batch["label"]       # float32 [N]，供 BCE 使用
        ids = batch["image_id"]       # 用于追溯与配对
        # TONG/WANG 在这里接入 forward、loss 与更新。
```

上例只展示数据接入，不是训练入口。`batch["label"]` 为 `[N]`；若模型输出为 `[N,1]`，训练方应统一形状，例如 `logits.squeeze(1)`，防止错误广播。将张量送到设备的工作由训练或推理方完成。

| 样本字段 | 类型或含义 |
|---|---|
| `image` | 经归一化的 `float32[3,224,224]` |
| `view` | 仅 B/C 的 `train` 或 `fixture` 样本返回；与 `image` 共享原图和几何处理 |
| `label` | float32 标量，0=Real，1=AI |
| `image_id` | 清单中的图片 ID |
| `generator`、`split`、`rel_path` | 原始清单元数据 |
| `transformation` | 基础套件名称；B/C 的随机模式及实际参数见 `view_params` |
| `view_params` | JSON 字符串形式的实际变换参数，可被默认 DataLoader 合并 |

验证和测试使用单视图、固定套件。创建验证/测试数据集时使用 `condition="A"`，以避免将训练 paired-view 逻辑用于选模型或测试。

```python
validation = GenImageDataset(
    "Images/manifest.csv", split="val", condition="A", suite="clean",
    config=config
)
test_jpeg = GenImageDataset(
    "Images/manifest.csv", split="test_unseen_ADM",
    condition="A", suite="jpeg_q70", config=config
)
fixture = GenImageDataset(
    "Images/fixture/manifest_fixture.csv", split="fixture",
    root="Images", condition="B", seed=5421, config=config
)
```

ZHUO 的单张图片入口应复用同一变换，不能另写缩放或归一化。以下只生成模型输入，模型恢复和概率输出接入其自己的 CLI。

```python
from PIL import Image
from src.datasets import EvalTransform

transform = EvalTransform(config=config, suite="clean")
with Image.open(image_path) as image:
    inputs = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
```

## 固定预处理与变换

所有 A/B/C 条件使用同一条共同预处理路径：按 EXIF 方向校正并转 RGB；以 bicubic 插值将短边缩放到 256，同时保持宽高比；中心裁剪为 224×224；在像素域施加所选扰动；转 float32 并除以 255；按 ImageNet 均值 `(0.485,0.456,0.406)` 和标准差 `(0.229,0.224,0.225)` 逐通道归一化。这里的 clean 指不再施加额外扰动，原始 JPEG 压缩或低分辨率痕迹仍然存在。

成对视图使用同一几何处理结果。干净分支为 `image`，变换分支为 `view`；不额外引入随机裁剪、翻转或颜色变换。变换模式在 JPEG、降采样、先降采样再 JPEG 三者中等概率选择。JPEG 质量为 50 到 95 的均匀整数；降采样比例在 0.5 到 0.9 间均匀采样，先缩小再恢复到 224×224。两类标签使用相同规则，参数采样不读取标签或生成器。

| 测试套件 | 固定处理 | 对 ZHANG 的交接要求 |
|---|---|---|
| `clean` | 仅共同预处理 | 每个测试划分的参考结果 |
| `jpeg_q70` | 224×224 像素域 JPEG q=70 后解码 | 和 clean 使用相同图片 ID |
| `jpeg_q50` | 224×224 像素域 JPEG q=50 后解码 | 和 clean 使用相同图片 ID |
| `downsample_0.5` | 224→112→224 | 固定缩放比例，不按原生尺寸缩放 |
| `gaussian_blur_1.0` | Pillow GaussianBlur，radius=1.0 | 记录实现和依赖版本 |
| `bias_control_jpeg95` | 原图转 RGB 后统一缩放到 256×256，JPEG q=95 解码，再进入共同预处理 | 作为额外偏差对照，单独报告 |

JPEG、降采样和模糊均作用于未归一化的 RGB 像素。偏差对照在内存中生成，不覆盖原图或修改原始哈希。它统一了新增的尺寸和编码处理，但不能消除此前 JPEG 压缩、原生分辨率及重采样历史的痕迹，因此不能据此声称偏置已被完全消除。

像素级复现细节：JPEG 使用 Pillow，`subsampling=2`、`optimize=False`、`progressive=False`；降采样与恢复均使用 bicubic，缩小后的各边长度取 `max(1, int(原边长 × 比例))`。共同 resize 的长边向下取整，中心 crop 起点使用 Python `round`。依赖版本应随运行保存，不要求不同 Pillow/JPEG 编解码库版本之间逐像素相同。

## B 与 C 的实验控制

同一个共享种子、epoch 和 `image_id` 确定同一组变换参数；规则不依赖条件 B/C、标签、worker 编号或读取顺序。B/C 必须同时使用相同清单顺序、数据配置、种子、epoch、batch 大小及 loader 参数，才能得到相同批次和视图。每个 epoch 开始前调用 `dataset.set_epoch(epoch)`，恢复训练时恢复正确 epoch。

数据模块提供相同输入的条件；训练方还需对齐初始化、批次大小、优化器更新次数、梯度累积、早停/预算规则和随机种子。两个条件若实际使用不同更新预算，不能仅凭相同数据变换声称完成受控消融。

为每次运行保存：代码提交号、Python/PyTorch/Pillow/NumPy 版本、完整数据配置及 `config.fingerprint`、`dataset.manifest_sha256`、训练种子、epoch、DataLoader 参数和数据 QA 报告。`config.fingerprint` 是规范化配置内容的 SHA256，不是配置文件原始字节的 SHA256。不能使用 Python 的随机化 `hash()` 替代稳定的逐图种子规则。

## 数据 QA 和正式训练门槛

原仓库 `Images/verification.json` 是 FEI 的已有记录，不等于本次机器上的接收核验。本次应分别记录清单核验、图片字节与解码核验、感知近重复候选及人工复核状态。缺图片、未运行感知检查或存在未复核跨划分候选时，不能把接收状态记为“全部通过”。

QA 工具核对正式清单和 fixture 的数量、标签/生成器关系、图片 ID、路径、SHA256、source MD5 及跨划分重叠。全量模式读取文件，检查解码、尺寸、格式、文件体积和实际哈希。pHash 用于发现跨划分的视觉相似候选，候选应人工查看，不能自动当成同一图片或自动删除。任何最终划分修正应由数据负责人确认，再同步清单版本、数据配置和训练台账。

pHash 固定采用灰度 32×32 LANCZOS、正交 DCT-II、左上 8×8 频率系数；阈值为不含 DC 的 63 个 AC 系数中位数，DC bit 置零；跨划分 Hamming 距离 ≤4 的图片对列为候选，包含 fixture 与正式集之间的比较。BK-tree 检索不按类别或标签预筛选。报告中计数完整，最多保留 10,000 个候选例子。该方法不能排除裁剪、旋转或语义近重复。退出码：元数据通过为 0（仍未核验图片）；错误为 1；缺图或未做 pHash 为 2；全量完成且有候选待复核为 3；仅全量完成且无候选才标记 `full_qa_pass=true`。

类别审计显示：训练集 real/AI 各覆盖 1,000 类，单类计数最多相差 1，类别分布总变差距离为 0.07833333；验证与三个测试集均为 0。fixture 的类别支持集不同，总变差距离为 0.88，故只用于开发联调，不能用于正式平衡评估，也不能据此估计 ImageNet 的自然类别先验。

当前接收与近重复核验状态以本次生成的 QA JSON 为准；这份方法说明不预先宣称零泄漏或零近重复。完整图片未就绪时，可用 fixture 完成接口联调，正式训练仍需完成上述接收核验和跨划分风险处理。

本次原始筛查保留 `review_required`，其唯一候选的判定保存在单独的 `qa_visual_review.json`；应结合 [`ouyang_qa.md`](ouyang_qa.md) 阅读，避免把自动候选误认为已确认重复。所有 QA 和试跑输出必须放在图片根目录外，防止错误输出参数覆盖原始数据。

## 各成员接入与剩余依赖

| 接收方 | 可以直接使用的内容 | 仍需该成员完成 |
|---|---|---|
| TONG | A 的 `image`、B 的 `image/view`、标签和固定配置 | ResNet-18、A/B 训练、checkpoint 与试跑记录 |
| WANG | 与 B 一致的 C 成对视图、epoch 规则和参数记录 | 一致性损失、六次训练预算与排期 |
| ZHANG | 三个测试划分的固定扰动和偏差对照、稳定图片 ID | 批量预测、指标与配对 bootstrap；按同一 ID 对齐套件 |
| ZHUO | 与评估同源的预处理和固定标签含义 | CLI 加载 checkpoint、输出 P(AI)、独立复核与展示整合 |
| FEI | QA 工具、风险说明和英文方法片段 | 近重复候选处置、数据版本确认和报告合并 |

正式评估使用 AI 为正类、阈值固定 0.5；只在验证集选择 checkpoint。三个测试集各报告 clean、q70、q50、0.5× 降采样和 blur，偏差对照另列。配对图像 bootstrap 反映图像采样差异，不涵盖全部训练不确定性。本交付不包含模型效果提升的结论，结果须由正式预测与指标支持。

报告合并时，`ouyang_methods.tex` 作为方法片段插入现有 CVPR 文档；`ouyang_presentation.md` 提供个人约 1.5 分钟讲稿，配合扰动样例即可使用。与训练成员共同试跑、完整模型测试和交叉复核的结果需在组内运行后补入台账。
