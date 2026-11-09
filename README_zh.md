# Supermimic-dex

这是把 [ConTrack](https://github.com/EmptyBlueBox/ConTrack) 的训练方式改为参考 [Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) 的 BPTT 后，整理出的 Hammer 最佳版本。

**已保存的 BPTT 策略成功率为 99.9756%（4095/4096）**，从头训练，`e3 / seed 2026`，原源码提交 `a232096726ae6c9a6523aabe3b8cd7a156bd66a0`。Hammer 使用双手 xArm7/XHand。权重、完整训练配置、评估记录和源码校验信息均在仓库内。

- [安装与环境](docs/SETUP.md)
- [英文 README：训练、评估、录视频的完整命令](README.md)
- [BPTT 与 WM 设计](docs/METHOD.md)
- [复现与版本管理说明](docs/REPRODUCIBILITY.md)
- [最佳训练配置](configs/hammer_best.json)
- [BPTT 权重](results/hammer_best/policy.pt)与[评估记录](results/hammer_best/evaluation.json)

整理保留了原 ConTrack 的真实环境输入输出、动作执行、奖励和成功判定。想象轨迹使用可微接触奖励近似和软 continuation；这些训练信号与真实评估奖励的区别在方法说明中明确列出。

算法核心冻结在最佳版本。后续失败头、CUDA Graph、小 Actor、PPO 初始化等实验没有混入本次发布。整理修改集中在代码格式、去除未使用导入、可配置路径、同仓库 PPO 入口、复现文档及校验脚本。

这里的 99.9756% 只代表一个选出的 Hammer 校准实验，不代表全任务或多个训练种子的平均结果；成功判定按物体位置和旋转阈值统计，Contact F1 为 0.69264。它也不能证明 BPTT 比 PPO 更快达到相同质量。

机器人资源与参考数据从固定的上游版本下载，本仓库不重新分发这些数据。使用 Hammer 的 GRAB 衍生参考数据需要遵守原始研究用途条款，详见[第三方说明](THIRD_PARTY_NOTICES.md)。
