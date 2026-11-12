# Superpod 正式 C 运行记录

2026-11-12，运行代码固定为 `b16b057ab82802da016639d605707fb482be72b1`，来自 WANG 分支。使用独立目录 `$HOME/arin5201-wang`，没有修改其他项目或全局Python环境。

| 作业 | 分区与资源 | 工作 |
| --- | --- | --- |
| 775721 | cpu，4CPU、16GB | 安装独立环境，接收全量数据，核验QA、缓存权重并运行CPU测试 |
| 775722 | normal，1GPU | 在准备成功后运行GPU测试、短试跑，然后依次训练C_5421和C_2026 |

GPU作业通过 `afterok:775721` 依赖准备作业，准备失败不会开始正式训练。作业提交不计作实验完成，最终状态与产物按实测补充到本页及 [六次运行台账](six-run-ledger.csv)。

CPU准备已开始。GPU类型和训练耗时以调度分配后的记录为准；当前尚无正式模型。配置和复查方式见 [正式训练设置](wang-training-protocol.md) 及 [Slurm脚本说明](../scripts/superpod/README.md)。
