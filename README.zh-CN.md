# osu! AI：像人一样打 osu!

用顶尖玩家的回放训练神经网络，学习怎么玩节奏游戏 [osu!](https://osu.ppy.sh)，再在 osu!lazer 里回放。
目标不是做一个完美的自动打图程序，而是让它打得**像人**：光标轨迹平滑但不完美，会真正地转转盘，按键节奏自然。

[English](README.md) · 论文：[中文 PDF](paper/paper_zh.pdf) · [English PDF](paper/paper_en.pdf) · 实验记录：[results/RESULTS.md](results/RESULTS.md)

![真人、LSTM 和 TCN-WGAN 在未见过谱面上的光标轨迹](docs/images/trajectories.png)

*在没见过的谱面上，LSTM 瞄得和 GAN 一样准，但转盘时停着不动（下排）；TCN-WGAN 像真人一样转。*

## 结果

用 1,992 份回放训练，在 50 张从未参与训练的谱面上测试：

| | Relax 准确率 | 完整准确率（含按键） | 转盘转速 |
|---|---|---|---|
| 真人回放 | 99.98% | 96.04% | 371 RPM |
| LSTM 基线 | 99.92% | 95.74% | 3 RPM |
| **TCN-WGAN** | **99.89%** | **95.45%** | **394 RPM** |

完整准确率使用按键模型加上按键合并（见下文）；不合并时 TCN-WGAN 为 93.55%。Relax 准确率假设按键时机完美，只衡量光标。

osu!lazer 实战（未登录状态）：

| 谱面 | 星级 | 模组 | 准确率 | 300 / 100 / 50 / Miss |
|---|---|---|---|---|
| Ado - AiAiA [Kiss me Insane] | 4.07 | | 98.38% | 668 / 12 / 2 / 3 |
| Ado - AiAiA [Be loved Expert] | 5.70 | | 95.85% | 820 / 39 / 15 / 3 |
| 米津玄師 - IRIS OUT [Darling!!!] | 7.11 | HD HR | **96.63%** | 654 / 28 / 0 / 5 |

最后一局按键时间的标准差为 8.1 ms（UR 81）。

## 主要发现

提升大多来自修正训练目标和评估方法，而不是换新的模型结构。论文从原理上解释了每一条：

1. **GAN 的判别器根本没学会。** 有限差分梯度惩罚只沿一个随机方向测梯度，在 4,096 维的轨迹里只测到约 1/64。改用卷积判别器上的标准梯度惩罚后解决。
2. **GAN 瞄不准**，因为学习率太低。先用更高的学习率预训练生成器，Relax 准确率从 56.5% 升到 96.7%。
3. **回归模型不会转盘。** 它学到的是平均轨迹，而所有玩家转盘画的圆平均起来就是圆心。为转盘单独设计极坐标输出，再加上约束转速和半径的损失，GAN 就会转了。
4. **原评估用的圆圈半径只有实际的一半。** 修正后，离线预测和实战结果相差约一个百分点。
5. **重复按键让实战掉分。** 按键模型常常两个键一起按，多出的按键会提前打到下一个物件，或者抢走滑条头后马上松开。把间隔小于 36 ms 的按下合并后，7.11 星那张图一个 50 都没有了。

![合并按键前后的按键模型输出](docs/images/key_decoding.png)

## 工作原理

```
.osu 谱面 ──► 特征（12 ms 一帧）──► TCN-WGAN 生成器 + 噪声 ──► 光标轨迹
                               └─► 按键 LSTM ──► 合并按键 ──► 按键
                                                                │
osu!lazer ◄── 鼠标和键盘 ◄── 回放循环 ◄── tosu 提供的歌曲时间 ◄──┘
```

| 模块 | 作用 |
|---|---|
| `osuai/beatmap.py`、`slider_path.py` | 读取 .osu 文件、滑条曲线、模组（HR、DT、HT、EZ） |
| `osuai/replay.py` | 读取 .osr 回放 |
| `osuai/features.py` | 12 ms 时间轴上的谱面特征和回放目标 |
| `osuai/models.py` | TCN-WGAN 生成器和判别器、LSTM 基线、按键模型 |
| `osuai/train.py` | 训练：预训练、标准梯度惩罚、转盘损失 |
| `osuai/decoding.py` | 按键合并 |
| `osuai/judge.py`、`evaluate.py` | 按游戏规则给模型和真人评分 |
| `osuai/dataset.py`、`sources/` | 生成训练集，导入回放和谱面 |
| `osuai/play/` | 通过 tosu 在 osu!lazer 中回放 |

## 运行方法

在 Windows 11 + NVIDIA 显卡上测试，使用 [uv](https://docs.astral.sh/uv/) 和 Python 3.11。

```bash
uv sync
```

**评估**仓库里的 V3 模型（需要 `data/` 里有回放和谱面）：

```bash
uv run python -m osuai.evaluate --split test --gaps none 36
```

**在 osu!lazer 中回放。** 启动 [tosu](https://github.com/tosuapp/tosu)（在它的配置里设 `POLL_RATE=10`），退出 osu!lazer 的登录，关闭"高精度鼠标"。
打开窗口，在游戏里选好谱面，按 **Q** 开始，**W** 停止，**R** 松开按键。勾选 "Cursor only" 后由 Relax 模组负责按键。只有在未登录状态下才会回放。

```bash
uv run python -m osuai.play
```

**获取数据。** 从 [osu!3k](https://www.kaggle.com/datasets/ilymeow/osu3k) 数据集导入（需要你自己的 Kaggle API 令牌），
或者从 osu! 官网下载某位玩家的回放（需要在 `.env` 文件里写入你自己的 `OSU_SESSION` cookie，git 会忽略这个文件）：

```bash
uv run --extra osu3k python -m osuai.sources.osu3k --count 1000
uv run python -m osuai.sources.osu_web --user 7562902
```

**训练：**

```bash
uv run python -m osuai.dataset --name v4
uv run python -m osuai.train cursor --data v4 --out models/v4 --epochs 90 --pretrain 60
uv run python -m osuai.train keys --data v4 --out models/v4 --epochs 30
```

## 使用须知

这是一个研究如何模仿人类行为的项目。所有游戏内测试都在未登录状态下进行，没有任何成绩被提交到线上。
在线上账号中使用生成的游玩违反 osu! 规则，对其他玩家不公平，也会导致封号。回放程序不包含任何反检测功能。

## 背景与致谢

项目的想法和几类模型（LSTM、VAE、WGAN、TCN-WGAN）来自 [niooii/osu](https://github.com/niooii/osu)，该项目又受到
[OsuLearn](https://github.com/GuiBrandt/OsuLearn) 的启发。本项目的第一版建立在 niooii 的代码之上，并用它训练了 V3 模型；
之后本仓库的所有代码都为本项目从头重写，V3 模型已转换到新代码，输出完全相同。
数据：[osu!3k](https://www.kaggle.com/datasets/ilymeow/osu3k) 数据集（MIT 许可）。回放：[tosu](https://github.com/tosuapp/tosu) 和 [osu!lazer](https://github.com/ppy/osu)。

本项目的代码、实验和论文在编写过程中使用了 Claude（Anthropic）作为编程助手。仓库不包含回放、谱面和音乐文件。
