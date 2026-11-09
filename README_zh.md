# Supermimic-dex

这是把 [ConTrack](https://github.com/EmptyBlueBox/ConTrack) 的训练方式改为参考 [Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) 的 BPTT 后，整理出的 Hammer 最佳版本。

**BPTT 策略已保存的成功率为 99.9756%（4095/4096）**，从头训练，`e3 / seed 2026`，原源码提交 `a232096726ae6c9a6523aabe3b8cd7a156bd66a0`。Hammer 使用双手 xArm7/XHand。权重、完整配置、评估记录及源码校验信息均在仓库内。

[观看 Hammer BPTT 视频：18 秒，连续 3 回合](docs/media/hammer_bptt_best.mp4) · [BPTT 权重](results/hammer_best/policy.pt) · [评估记录](results/hammer_best/evaluation.json)


[观看 6 段 BPTT 任务视频](docs/SUCCESSFUL_VIDEOS.md)：Hammer、酒杯、水瓶、华夫饼机，以及 XHand、Sharpa 两种机械手的方块交接。每段新视频包含 3 个筛选后的成功回合，并附独立完整成功率和录制记录。

## 跨任务更新：2026-11-09

新增 10 个 XHand 任务的 BPTT/PPO 对照已完成。使用相同 Hammer e3 配置，从头训练的 **酒杯成功率为 99.9756%**，**水瓶为 99.6338%**；两份 BPTT 权重及评估记录已包含在仓库中。[完整结果表、数据链接和复现命令](docs/CROSS_TASK_RESULTS.md)。

这轮是单 seed、每项 64M 交互的结果。其他任务仍明显落后于 PPO，实测 BPTT 训练耗时为 PPO 的 1.51–1.86 倍。13 个 Sharpa 任务的对照已结束，[结果见这里](docs/SHARPA_RESULTS.md)。`v0.1.0` 保留原 Hammer 发布版本；获取新权重请克隆当前 main。

## 1. 克隆代码

```bash
git clone --branch v0.1.0 https://github.com/Hxxxz0/Supermimic-dex.git
cd Supermimic-dex
```

## 2. 数据在哪里，怎么下载

复现用的是 ConTrack **已经完成机器人重定向的 HDF5 参考片段**，包含双手机器人轨迹、物体几何与位姿、接触标注。复现这个版本不用重新重定向原始 GRAB 数据。

| 文件 | 下载链接 | 用途 |
| --- | --- | --- |
| Hammer / xArm-XHand | [Hammer HDF5](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5) | 本次最佳 BPTT 的训练、评估与视频 |
| Cylinder / xArm-XHand | [XHand 默认 HDF5](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/xhand/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5) | 上游任务注册所需的默认数据 |
| Cylinder / RB-Y1A-Sharpa | [Sharpa 默认 HDF5](https://raw.githubusercontent.com/EmptyBlueBox/ConTrack/f5efe18e4b1f986566b252265c4bf63f3b67c6c6/data/sharpa/dexterhand-Cylinder_00-fps_20-rby1a_sharpa-3600_3700.h5) | 上游任务注册所需的默认数据 |

推荐一键下载数据与机器人资源：

```bash
python3 scripts/supermimic/fetch_assets.py
python3 scripts/supermimic/fetch_assets.py --verify-only
```

脚本从[固定版本的 ConTrack 压缩包](https://github.com/EmptyBlueBox/ConTrack/archive/f5efe18e4b1f986566b252265c4bf63f3b67c6c6.tar.gz)获取 189 个必要文件，并逐一校验 SHA256 和字节数。数据位于 `data/xhand`、`data/sharpa`，机器人文件位于 `assets/meshes`、`assets/urdf`。只下载 Hammer 文件还不够：机器人资源及两份注册默认数据也需要准备。

原始数据来源：[GRAB](https://grab.is.tue.mpg.de/) · [DexterHand](https://huggingface.co/datasets/pku-mocca/DexterHand)。Hammer 是 GRAB 的衍生参考数据，需要遵守原始非商业科研用途条款，详见[第三方说明](THIRD_PARTY_NOTICES.md)。本仓库不重新分发或重新许可这些资源。

## 3. 配置环境

| 组件 | 验证过的版本 |
| --- | --- |
| Python | 3.11 |
| Isaac Sim | 5.1.0，具体 native build 见原实验记录 |
| Isaac Lab | 2.3.2，提交 `42e61645c96bac08135566634785cdc87728d5ab` |
| PyTorch | 2.7.0 + CUDA 12.8 |
| RSL RL / tensordict | 3.1.2 / 0.11.0 |
| 原实验 GPU | 单张 RTX 5090，报告显存 32607 MiB |

### 已有 Isaac 镜像

使用镜像自带的 Python，将 Isaac Lab 路径替换为实际固定版本的 checkout。尚未准备 Lab 时，先克隆并切到上面的提交。

```bash
export CONTRACK_ISAACLAB_PATH=/absolute/path/to/IsaacLab
export CONTRACK_RESULTS_DIR="$PWD/outputs"
# Isaac 不在默认位置时才需要设置：
# export ISAAC_SIM_PYTHON=/absolute/path/to/isaac-sim/python.sh
bash scripts/supermimic/python.sh -c 'import torch, isaaclab; print(torch.__version__)'
bash scripts/supermimic/python.sh -m pip install --no-deps -e .
```

### 从零安装

在已经克隆的本仓库目录运行：

```bash
conda create -n supermimic_dex python=3.11 -y
conda activate supermimic_dex
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
git -C ../IsaacLab checkout 42e61645c96bac08135566634785cdc87728d5ab
python -m pip install 'setuptools>=77,<81' wheel
python -m pip install --no-build-isolation flatdict==4.0.1
(cd ../IsaacLab && ./isaaclab.sh --install rsl-rl)
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e source/ConTrack
export CONTRACK_ISAACLAB_PATH="$(realpath ../IsaacLab)"
export CONTRACK_RESULTS_DIR="$PWD/outputs"
```

### 生成 USD 并校验

```bash
python3 scripts/supermimic/fetch_assets.py
bash scripts/supermimic/python.sh scripts/supermimic/prepare_assets.py --device cpu
python3 scripts/supermimic/verify_release.py
```

下载资源与生成文件均由 `.gitignore` 排除，实验输出写入 `CONTRACK_RESULTS_DIR`，正式训练要求源码提交干净。更多环境细节见[安装说明](docs/SETUP.md)。

## 4. 训练、评估、录视频

下面三个命令都使用同一个 Hammer 片段。训练 JSON 是完整配置的 overrides，preset 由 `--preset e3` 单独指定；校验脚本会检查解析后的配置与原实验完全一致。

```bash
# 从头训练最佳版本：8000 环境，64M 真实 transition。
bash scripts/supermimic/python.sh scripts/supermimic/train.py \
  --algorithm supermimic --preset e3 --seed 2026 --budget-updates 500 \
  --config configs/hammer_best.json --num-envs 8000 --device cuda:0 \
  --data data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5

# 用仓库内的 BPTT 权重执行原协议的 4096 回合评估。
bash scripts/supermimic/python.sh scripts/supermimic/evaluate.py \
  --checkpoint results/hammer_best/policy.pt \
  --data data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5 \
  --episodes 4096 --num-envs 256 --seed 12345 --device cuda:0 \
  --output "$CONTRACK_RESULTS_DIR/hammer-evaluation.json"

# 录制连续 3 回合的 BPTT 策略推理，不筛选成功回合。
bash scripts/supermimic/python.sh scripts/supermimic/record_video.py \
  --checkpoint results/hammer_best/policy.pt \
  --data data/xhand/grab-s1_hammer_use_1-xarm7_xhand-120_840.h5 \
  --episodes 3 --device cuda:0 --output "$CONTRACK_RESULTS_DIR/hammer-bptt.mp4"
```

## 方法与结果边界

整理保留了原 ConTrack 的真实环境输入输出、动作执行、奖励及成功判定。想象轨迹使用可微接触奖励近似和软 continuation，详见[BPTT/WM 设计](docs/METHOD.md)。

训练核心冻结在最佳版本，后续失败头、CUDA Graph、小 Actor、PPO 初始化等实验没有混入发布。整理集中在格式、去除未使用导入、可配置路径、同仓库 PPO 入口、复现文档及校验脚本，详见[复现说明](docs/REPRODUCIBILITY.md)。

99.9756% 只代表一个选出的 Hammer 校准实验，不代表全任务或多个训练种子的平均结果。成功按物体位置 0.1 m、旋转 1 rad 阈值统计，Contact F1 为 0.69264；它也不能证明 BPTT 比 PPO 更快达到相同质量。原实验从头训练，整理后没有重新跑完整 64M 训练。 整理后的独立 Git checkout 已用同一份 BPTT 权重重新完成 4096 回合评估，成功率、Progress、物体位姿误差及 Contact F1 与原记录完全一致，详见[整理后回放验证](results/hammer_best/evaluation_clean_release.json)。
