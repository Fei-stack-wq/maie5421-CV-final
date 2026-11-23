# Superpod 上运行两个 C 实验

使用项目独立 checkout 和 `.venv`。先在 `cpu` 分区准备环境与数据，再在 `normal` 分区申请1张GPU，依次运行C_5421和C_2026；账号和QOS从本人实际授权中填写，不沿用别人的账户。

以下命令从服务器上的项目根目录执行。每次正式运行必须使用固定代码提交和空输出目录，当前脚本不自动覆盖或重跑已有 `runs/C_<seed>`。

```bash
module load slurm/slurm/23.02.6
mkdir -p runs
prepare_job=$(sbatch --parsable --account=<course-account> --qos=cpu_qos \
  --output="$PWD/runs/prepare-%j.out" scripts/superpod/prepare.sbatch "$PWD")
sbatch --account=<course-account> --qos=<authorized-gpu-qos> \
  --dependency=afterok:${prepare_job} --kill-on-invalid-dep=yes \
  --output="$PWD/runs/formal-c-%j.out" scripts/superpod/train_c.sbatch "$PWD"
```

准备作业会验证12100张图片的字节、解码和pHash，比较现有候选处置记录。原始筛查JSON保持原状态；单独的 `superpod-data-ready.json` 说明新核验与既有复核一致，不冒称新增人类签字。出现新候选、文件错误或不完整覆盖时停止，不启动训练。

GPU作业先运行测试及OUYANG变换的CUDA小样本检查，再使用 [两个正式配置](../../configs/wang/) 训练。依赖作业失败时，不进入正式训练。调度器会保存运行日志，代码会保存每轮更新数、AMP跳步、最佳模型和恢复点。

模型及完整日志保留在服务器的 `runs/`，不上传Git。checkpoint交付时保留配置、history和文件哈希；单独推理使用best.pt，恢复需要last.pt及其指向的best_epoch文件。

资源和存储规则参考学校的 [Slurm说明](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/slurm)、[学生账号说明](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/account-for-teaching-course) 和 [存储说明](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/fileandstorage)。本项目使用home中的独立目录，CPU与GPU节点均可访问；不在登录节点运行训练。
