# Images/ — GenImage 12k 子集（v1.0）

本目录是 Project 6「AI 生成图像检测」的**唯一数据集**：12,000 张已按 train / val / test 划分好的图像，
外加 1 份来源清单（manifest.csv）、1 份 100 张开发样例（fixture）与 1 张预览图。

## 1. 目录结构

```
Images/
├── train/                      # 6,000 张：3,000 real + 1,500 BigGAN + 1,500 SD1.4
│   ├── real/*.JPEG
│   ├── BigGAN/*.PNG
│   └── stable_diffusion_v_1_4/*.PNG
├── val/                        # 1,500 张：750 real + 375 BigGAN + 375 SD1.4（调参 / 早停 / 选阈值）
│   ├── real/
│   ├── BigGAN/
│   └── stable_diffusion_v_1_4/
├── test_seen/                  # 1,500 张：750 real + 375 BigGAN + 375 SD1.4（已见生成器测试）
├── test_unseen_ADM/            # 1,500 张：750 real + 750 ADM（留出生成器，只在测试出现）
├── test_unseen_GLIDE/          # 1,500 张：750 real + 750 GLIDE（留出生成器）
├── fixture/                    # 100 张开发样例（50 real + 25 BigGAN + 25 SD1.4，94 类），不参与正式划分
│   ├── real/  BigGAN/  stable_diffusion_v_1_4/
│   └── manifest_fixture.csv
├── manifest.csv                # 12,000 张的逐图来源记录（唯一权威清单）
├── dataset_summary.json        # 数量 / 类别 / 格式 / 分辨率汇总
├── verification.json           # 独立校验报告（0 error）
├── bias_audit.json             # 偏置审计表（格式、分辨率、每类数量）
└── preview_contact_sheet.png   # 每类抽样预览
```

> 文件名规则：`{ImageNet类别ID:03d}_{源文件md5前10位}.{扩展名}`，扩展名保留上游原始大小写（real 为 `.JPEG`，伪造图为 `.PNG`）。文件字节与 GenImage 上游**完全一致**，未重编码。

## 2. 标签约定

| 目录                                    | label | 说明                                         |
| --------------------------------------- | ----- | -------------------------------------------- |
| `real`                                | 0     | ImageNet 真实照片（JPEG）                    |
| `BigGAN` / `stable_diffusion_v_1_4` | 1     | 已见生成器伪造图（训练中出现过）             |
| `ADM` / `GLIDE`                     | 1     | **留出生成器**伪造图（训练中从未出现） |

AI = 正类（label 1），与提案 §5 一致。

## 3. 读取方式

不要用 `torchvision.datasets.ImageFolder` 直接指到 `Images/`（它会把 split 目录当成类别）。
请以 `manifest.csv` 为准：

```python
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class GenImageSubset(Dataset):
    def __init__(self, manifest, split, root="Images"):
        self.df = manifest[manifest.split == split].reset_index(drop=True)
        self.root = root
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r.rel_path}").convert("RGB")
        return img, int(r.label), r.image_id

manifest = pd.read_csv("Images/manifest.csv")
train_ds = GenImageSubset(manifest, "train")
```

设计实验时请按 `split` 取：`train` 训练、`val` 早停/选阈值、五个 test 目录只做最终评测。

## 4. 每条记录的来源字段

`source_repo` / `source_split` / `source_image_path` 指向上游 GenImage 子集的位置；
`source_md5` 是上游文件字节 md5（本数据集全部逐张比对通过），`sha256` 是本地文件哈希。
完整字段说明见 [docs/dataset-card.md](../docs/dataset-card.md)。

## 5. 版本控制说明

数据集本体（1.87 GB 图像与 100 张 fixture）**默认不进 git**（见仓库根目录 `.gitignore`），
仓库里只保留 `manifest.csv` / `dataset_summary.json` / `verification.json` / `bias_audit.json` /
`preview_contact_sheet.png` / `fixture/manifest_fixture.csv`。
共享数据请直接拷贝 `Images/` 目录（或用下面的命令重建），提交报告时把数据作为附件打包。

## 6. 重建

```bash
python tools/dataset/download_shards.py   # 下载所需 parquet 分片（约 16 GB 缓存）
python tools/dataset/build_dataset.py     # 按 seed=5421 抽样并写出 Images/
python tools/dataset/verify_dataset.py    # 独立校验
```
