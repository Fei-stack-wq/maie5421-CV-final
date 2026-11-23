# 欧阳迪数据模块验收记录

日期：2026 年 11 月 10 日。负责人：OUYANG Di 欧阳迪。基准仓库提交：`e7db18d0657865d5f344632534a3248cb49a6355`。

数据接口、固定变换、QA 工具及报告内容已交付。正式数据和 fixture 共 12,100 张已在本机完整接收，文件核验全部通过。跨划分 pHash 筛查产生 1 组候选，经 Codex 视觉检查判为不同内容，未修改任何源图片或划分。训练主流程联调及模型效果仍需 TONG/WANG/ZHANG 的入口和结果。

## 实测结果

| 项目 | 结果 | 证据 |
|---|---|---|
| 清单核验 | 正式 12,000 行及 fixture 100 行；划分、标签、数量、类别支持均符合约定 | `artifacts/ouyang/qa_metadata.json` |
| 精确重复与留出隔离 | image ID、路径、source MD5、SHA256 无重复；ADM/GLIDE 未进入训练或验证 | 同上 |
| 文件接收 | 新接收 12,000 张，已有 fixture 100 张；全部 SHA256 一致，0 下载错误 | `artifacts/ouyang/receipt_download.json` |
| 全量文件核验 | 12,100/12,100 张可解码；尺寸、格式、体积、SHA256 和 source MD5 全部符合清单 | `artifacts/ouyang/qa_received.json` |
| 跨划分感知筛查 | 12,100 张全部计算 pHash；Hamming 距离 ≤4 共 1 对 | 同上 |
| 候选视觉复核 | 1 对判为不同图像；0 对确认重复，0 对未复核；复核者为 Codex，非人类签字 | `artifacts/ouyang/qa_visual_review.json` |
| 接口与保护检查 | 50 项测试通过，0 失败，0 跳过 | `artifacts/ouyang/tests.xml`、`test_summary.json` |
| fixture 实图试跑 | 100 张通过 A/B 干净分支一致、B/C 视图一致、输出形状及有限值检查 | `artifacts/ouyang/smoke/smoke.json` |
| Windows 多进程 | 2 个 worker 下 fixture 100 张试跑通过；persistent worker 的 epoch 更新通过单元测试 | `artifacts/ouyang/smoke_workers2/smoke.json` |

## 唯一近重复候选的处置

- 训练图片：`train-72c90cdf3fa7`，`Images/train/BigGAN/021_72c90cdf3f.PNG`。
- 已见测试图片：`test_seen-7dbe0a810000`，`Images/test_seen/BigGAN/417_7dbe0a8100.PNG`。
- pHash 距离为 4。训练图是浅蓝天空中的小型深色鸟形主体；测试图是蓝灰天空中的彩色气球状主体。主体形状、颜色和布局不同，视觉上不符合单张图经缩放或重编码产生副本的情况，相似的简单背景可解释低频哈希接近。
- 判为筛查误报，保留两张图和原始划分。对照图为 [`near_duplicate_review.png`](../artifacts/ouyang/near_duplicate_review.png)。

原始筛查 JSON 保留 `status=review_required`、`full_qa_pass=false` 及退出码 3，不改写自动筛查证据。最终阅读时应同时查看视觉复核记录；其中记录清单与原始 QA 文件哈希、两张图片哈希、判定及依据。该复核没有冒称组员人工确认。pHash 筛查也不能排除所有裁剪、旋转或语义近重复。

## 固定版本与偏差说明

正式清单 SHA256：`265cdc4036036058666479cf2d717ed6eb2a7cdb6c7554109819b9c9280fa5f5`。

数据配置指纹：`a5618523e8077bdb917d2ed63b483e20487014f2e5b59a2e7195f6d21622bbae`。配置文件是 `configs/ouyang_data_v1.json`，代码与文件哈希清单见 `artifacts/ouyang/source_files.json`；实测依赖版本见 `artifacts/ouyang/environment.json`。

训练集两类均覆盖 1,000 个类别，单类计数最多相差 1，类别分布总变差距离为 0.07833333；验证及三个测试集为 0。fixture 的总变差距离为 0.88，仅用于开发，不能代替正式评估。统一重采样和 JPEG q95 对照用于检查格式与尺寸偏置，不能抹去原始图已有的压缩或低分辨率痕迹。

## 下游交接

TONG/WANG 可按 [`ouyang_handoff.md`](ouyang_handoff.md) 接入数据、成对视图和共享 epoch 规则；ZHANG 使用固定测试套件和图片 ID；ZHUO 复用单图 `EvalTransform`。报告方法片段见 [`ouyang_methods.tex`](ouyang_methods.tex)，约 90 秒展示内容见 [`ouyang_presentation.md`](ouyang_presentation.md)。

本次验收覆盖数据模块，不包含六次模型训练、checkpoint 恢复、正式预测或模型指标。组内完整试跑与后续实验结论应分别补入训练和评估台账。
