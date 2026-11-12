# 正式训练设置

2026-11-11，WANG 同意 TONG 在交接清单中提出的训练参数。2026-11-12，WANG 指定在 HKUST Superpod 执行本人两个 C 实验。A/B 的执行安排仍由 TONG 对接，不将已有 pilot 计入正式六次实验。

| 项目 | 设置 |
| --- | --- |
| 训练种子 | 5421、2026，A/B/C 共享 |
| 模型 | ImageNet 预训练 ResNet-18，只训练 layer4 和二分类头 |
| BN | 所有 running statistics 冻结；layer4 的 affine 参数可训练 |
| 数据 | OUYANG v1，`configs/ouyang_data_v1.json`，clean validation |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01 |
| 批次 | microbatch=4，累积4次，有效 batch=16 |
| 预算 | 最多8 epoch；clean-val AUC选checkpoint；patience=2，min_delta=0 |
| 损失 | B使用双视图平均BCE；C在其上加lambda=1的Bernoulli JS |
| 推理 | AI=1，阈值0.5 |

C 的 AMP 初始 scale 使用本分支实测通过的1024；这是 WANG 的 GPU 配置，TONG 原分支没有该配置字段。A/B 使用本分支时应显式对齐；共同记录实际优化器更新、AMP跳步及早停，不能仅按epoch上限判断预算一致。

正式C配置：[c_5421.yaml](../configs/wang/c_5421.yaml)、[c_2026.yaml](../configs/wang/c_2026.yaml)。`protocol_confirmed: true` 记录上述参数已获 WANG 确认，不代表服务器数据接收核验或训练已经完成。

B/C共用样本顺序、逐图视图生成规则与最大更新预算。数据配置指纹为 `a5618523e8077bdb917d2ed63b483e20487014f2e5b59a2e7195f6d21622bbae`；每次正式运行仍保存实际清单哈希、源码提交、完整配置和库版本。

目前两次C仍未启动。服务器GPU型号、作业分区和运行环境以实际检查及调度分配为准；按集群调度规则申请计算资源，不在登录节点执行训练。完成服务器数据核验与GPU试跑后再启动正式配置。
