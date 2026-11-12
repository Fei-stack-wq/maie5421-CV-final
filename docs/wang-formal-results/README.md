# 两个正式 C 模型结果与交接

2026-11-12，两个正式C实验及独立checkpoint核验完成。使用Superpod NVIDIA H800 80GB，训练源码固定为b16b057；每次使用6000张训练图和1500张clean验证图。

| 种子 | 完成epoch | 最佳轮次 | 最佳clean-val AUC | 实际更新 | AMP跳步 |
| --- | --- | --- | --- | --- | --- |
| 5421 | 8 | 第6轮 | 0.9648533333 | 3000 | 0 |
| 2026 | 8 | 第6轮 | 0.9612248889 | 3000 | 0 |

checkpoint中epoch从0计，最佳epoch均为5。两次在第8轮达到patience=2，且已到8轮上限。上述是验证集选模型的指标，不是测试集或鲁棒性结论。

训练作业775722耗时19分01秒，含GPU测试和短试跑；独立核验775793耗时45秒，两个作业均退出码0。核验重新加载模型、核对配置和预处理，复现各1500张验证图的概率，最大误差均为0。

## 权重位置

```text
$HOME/arin5201-wang/runs/C_5421/best.pt
$HOME/arin5201-wang/runs/C_2026/best.pt
```

最佳权重SHA256：

- C_5421：`5f07fbcc7ffdf9126b6c3338f109479c298c24be283cde35c838d98a52d9ec7c`
- C_2026：`958b0f820bb205954370cf87f0aa011289c05b2efed4d090b00176140c612f4c`

配置、history、完整报告与逐图验证预测保存在服务器对应run目录，并已取回本地备份，未公开上传。推理使用best.pt；恢复还需last.pt及对应的best_epoch_005.pt。

ZHANG/ZHUO可使用src.training.checkpoint.load_model恢复，再按checkpoint里的preprocessing.data_config构建OUYANG的DataConfig和EvalTransform。阈值固定0.5，AI为正类。权重和完整记录另行交接；后续测试、扰动和配对统计由评估环节完成。
