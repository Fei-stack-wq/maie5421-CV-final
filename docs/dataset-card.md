# GenImage 12k 子集 · 数据集说明（Dataset Card）

| 项 | 值 |
|---|---|
| 版本 | v1.0（2026-11-03） |
| 维护人 | FEI, Wenxiang（数据与合规） |
| 位置 | `Images/`（12,000 张 + manifest + 校验/审计报告） |
| 体量 | 12,000 张 / 1,934 MB |
| 上游 | GenImage [1] → 子集镜像 `naatee/genimage-subset-9`（Hugging Face） |
| 许可 | CC BY-NC-SA 4.0（沿用上游），仅限课程非商业使用 |
| 复现 | `tools/dataset/` 下 4 个脚本，seed = 5421 |

---

## 1. 来源与合规

**层级链**：ImageNet（real）/ BigGAN、Stable Diffusion v1.4、ADM、GLIDE（fake）
→ GenImage (NeurIPS 2023 D&B) → `naatee/genimage-subset-9` → 本项目 12k 子集。

- 选用镜像而非官方仓库的原因：官方 GenImage 全集约 678 GB 且为 gated 仓库（本环境无 HF token），
  镜像保留了**上游原始文件字节**与**上游相对路径**，可逐张回溯。
- 镜像逐图提供：`image_path`（上游相对路径）、`md5`、`label`、`generator`、`family`、`class_id/class_name`、`width/height/format`、冻结的 `split`。
- **未做任何重编码或缩放**：`sha256`（本地）与 `source_md5`（上游）逐张比对，12,000/12,000 通过。
  这样才保住了低层取证痕迹（JPEG 压缩表、PNG 编码差异），避免制造"数据集自身引入的伪信号"。
- 生成器分配：**BigGAN + SD v1.4 = 已见生成器**（训练中出现），**ADM + GLIDE = 留出生成器**（只在测试出现），与提案 §2 一致。
- 本项目不训练任何生成器，也不调用付费生成 API。

> 说明：镜像的上游数据全部来自 GenImage 的 `train` 分区，其 `train/val/test` 为按固定种子、
  分层抽样的**冻结重划分**（60/10/30），并非 GenImage 官方 train/val 目录。本数据集**继承该冻结划分**
  （见 §3），因此"官方源划分"这一点与提案措辞有差异，需在报告 §2 中如实说明。

## 2. 划分与数量

| split | 用途 | real | BigGAN(seen) | SD1.4(seen) | ADM(unseen) | GLIDE(unseen) | 合计 | 类别数 |
|---|---|---|---|---|---|---|---|---|
| `train` | 训练（A/B/C 三条件） | 3,000 | 1,500 | 1,500 | — | — | **6,000** | 1,000 |
| `val` | 早停 / 选阈值 / 校准 | 750 | 375 | 375 | — | — | **1,500** | 375 |
| `test_seen` | 已见生成器测试 | 750 | 375 | 375 | — | — | **1,500** | 375 |
| `test_unseen_ADM` | 留出生成器测试 | 750 | — | — | 750 | — | **1,500** | 750 |
| `test_unseen_GLIDE` | 留出生成器测试 | 750 | — | — | — | 750 | **1,500** | 750 |
| 合计 | | 6,000 | 2,250 | 2,250 | 750 | 750 | **12,000** | |

另有 `Images/fixture/`：100 张开发样例（50 real + 25 BigGAN + 25 SD1.4，94 个类别），
从 train 池抽取但**不在 12,000 张之内**，供 OUYANG/TONG/ZHANG 在数据管道未就绪时并行开发。

**泄漏控制（已实测，非仅声明）**：

| 检查项 | 结果 |
|---|---|
| 12,000 张内重复文件（source_md5 / sha256） | 0 |
| 同一张图跨不同 split 出现 | 0 |
| 留出生成器（ADM / GLIDE）出现在 train / val / test_seen | 0 |
| 真实图跨 split 复用 | 0（6,000 张全不相交） |
| 全部文件可解码、尺寸/格式与 manifest 一致 | 12,000/12,000 |

> 真实图**全局唯一池**分配：train 3,000 / val 750 / test_seen 750 / ADM-test 750 / GLIDE-test 750，
> 五个集合互不相交；每个 split 内 real 与 fake 共享同一组类别，保证「内容类别」不是捷径。

## 3. 抽样规则（seed = 5421，可完全复现）

1. **按 split 取池**：从镜像冻结的 `train / val / test` 行中取（`source_split` 字段）。
2. **按 ImageNet 类别分层**：train 覆盖 1,000 类（每类 3 real + 1~2 BigGAN + 1~2 SD1.4 = 5~7 张）；
   val / test_seen 覆盖 375 类（每类 4 张）；两个 unseen 测试各覆盖 750 类（每类 2 张）。
3. **real 与 fake 类别集合对齐**：先选类别集合，再在集合内分配 real 与 fake 数量，
   使每个 split 内 real/fake 的类别分布一致（$le 1$ 的差异），避免类别先验泄漏标签。
4. **真实图去重**：全局 `md5` 集合约束，任何一张真实图只进一个集合。
5. **随机性**：Python `random.Random(5421)` 顺序消费，样本可逐位复现（无需保存随机状态）。

## 4. manifest.csv 字段

| 字段 | 含义 |
|---|---|
| `image_id` | 唯一 ID：`{split}-{source_md5前12位}` |
| `split` | `train` / `val` / `test_seen` / `test_unseen_ADM` / `test_unseen_GLIDE` |
| `label` | 0 = real，1 = AI 生成 |
| `generator` | `real` / `BigGAN` / `stable_diffusion_v_1_4` / `ADM` / `GLIDE` |
| `family` | `real` / `GAN` / `diffusion` |
| `class_id` / `class_name` | ImageNet 类别（0–999 与其名称） |
| `filename` / `rel_path` | 相对 `Images/` 的路径 |
| `source_repo` | 上游数据集 ID |
| `source_split` | 上游冻结 split（train/val/test） |
| `source_image_path` | 上游相对路径，如 `BigGAN/train/ai/010_biggan_00157.png` |
| `source_md5` | 上游文件字节 md5（已逐张校验） |
| `sha256` | 本地文件哈希（团队共享接口要求的字段） |
| `width` / `height` / `format` / `bytes_source` / `bytes_file` | 尺寸 / 格式 / 体积 |
| `selection_seed` | 抽样种子 |

## 5. 偏置审计（本数据集最重要的一条结论）

| 生成器 | n | 格式 | 宽度 min/中位/max | 高度 min/中位/max | 平均体积 |
|---|---|---|---|---|---|
| real | 6,000 | JPEG | 38 / 500 / 3,648 | 38 / 375 / 3,485 | 111.8 KB |
| BigGAN | 2,250 | PNG | 128 / 128 / 128 | 128 / 128 / 128 | 28.9 KB |
| SD v1.4 | 2,250 | PNG | 512 / 512 / 512 | 512 / 512 / 512 | 448.0 KB |
| ADM | 750 | PNG | 256 / 256 / 256 | 256 / 256 / 256 | 116.4 KB |
| GLIDE | 750 | PNG | 256 / 256 / 256 | 256 / 256 / 256 | 76.5 KB |

**风险（必须在报告中显式处理，否则结论不成立）**：

1. **格式捷径**：real 全为 JPEG、fake 全为 PNG，分类器可能学到容器格式而非生成痕迹。
2. **分辨率捷径**：real 分辨率连续分布（38–3,648），fake 为固定 128 / 256 / 512，
   仅凭"原生分辨率是否非整数倍"即可获得很高准确率。
3. **上采样模糊**：BigGAN 128×128 升到 224×224 的模糊程度显著高于 SD 512×512 的降采样，
   该差异与"生成器类型"高度共线，会污染 A/B/C 条件之间的比较。

**建议（按优先级）**：

- 训练/评测统一做 `resize(256) → center_crop(224)` 或 `resize(224)`，并在 config 中固定；
  这一步必须在 A/B/C 三条件中**完全一致**（提案 §3 已要求）。
- 报告**必做对照实验**：只保留 `train/` 的 1,000 类里"real 与 fake 原生尺寸最接近"的子集
  （或把全部图像重采样到 256×256 后重编码为 JPEG q=95 生成一个对照变体），比较指标下降幅度，
  以说明模型是否依赖分辨率/格式捷径。对照变体可由未来脚本 `make_bias_controlled.py` 生成，
  本版本不默认提供（会破坏原始字节取证链）。
- 报告偏置结论时区分"跨生成器泛化下降"与"分辨率域偏移"，不要把后者算作一致性正则的功劳。

## 6. 校验结果

`Images/verification.json`：`ok = true`，0 error。

- manifest 行数 12,000；13 个「split × generator」组合数量与计划完全一致；
- 12,000 个文件全部用 PIL 解码成功，尺寸与 `format` 与 manifest 一致；
- `sha256` 逐文件复核一致；`source_md5` 无重复；跨 split 无共享图像；
- 留出生成器未泄漏到训练/验证/已见测试；
- 类别数：train 1,000、val 375、test_seen 375、ADM 750、GLIDE 750。

## 7. 复现步骤

```bash
# 环境：Python 3.12 + pyarrow（见 tools/dataset/requirements.txt 说明）
python tools/dataset/probe_shards.py     # 读取 90 个分片的 footer，建立分片→生成器映射
python tools/dataset/download_shards.py  # 下载 31 个分片（约 16 GB，落盘 tools/dataset/cache/shards）
python tools/dataset/build_dataset.py    # seed=5421 抽样 + 逐张 md5 校验 + 写出 Images/
python tools/dataset/verify_dataset.py   # 独立校验（有错即非零退出）
python tools/dataset/audit.py            # 偏置审计表
```

分片缓存可删除（`Remove-Item -Recurse tools/dataset/cache/shards`），重建时重新下载即可。

> **版本控制**：图像本体与分片缓存均被 `.gitignore` 排除，仓库只保留清单与脚本；
> 若课程要求随仓库提交数据，注释掉 `.gitignore` 中「数据集本体」一节即可（注意 GitHub 单文件 100 MB 限制）。

## 8. 许可与引用

- 数据许可：CC BY-NC-SA 4.0（上游）；ImageNet 图像仅限研究用途。
- 引用：Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023
  （github.com/GenImage-Dataset/GenImage）。

## 9. 与提案的差异（需在报告中如实说明）

| 提案原文 | 实际实现 | 影响 |
|---|---|---|
| 遵循 GenImage 官方源划分 | 使用镜像的冻结重划分（同样由官方 train 分区派生） | 划分自洽、无泄漏；但与官方 val 目录不同，需在 §2 说明 |
| 从训练集划出验证集 | 等价：val 与 train 来自同一上游 train 分区且互不相交 | 无 |
| 12,000 张、4 个生成器、全部集合互不相交 | 完全满足 | 无 |
| 近重复检测（near-duplicate） | 只做了精确 md5 去重；**未做** pHash/感知近重复 | 已在 §5 列为待办，W2 前补 |
| 内容类别分布配平 | 已按 ImageNet 类别分层，real/fake 共享类别集合 | 满足 |

## 10. 待办（W2 前）

- [ ] 感知近重复检测（pHash / DINO embedding），阈值与去重前后数量对账；
- [ ] 内容类别的 ImageNet 先验分布审计（当前按类均匀抽样，与 ImageNet 自然分布不同）；
- [ ] 可选：`make_bias_controlled.py` 生成 256×256 / JPEG q=95 对照变体，用于 §5 的捷径对照实验。
