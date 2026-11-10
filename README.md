# P1 β-VAE（二值化 digits）

在二值化的 sklearn digits 上比较 β ∈ {0, 0.1, 1, 4}。Encoder 为 64→128→(μ, logσ²)，潜变量维度 2；Decoder 为 2→128→64 logits。损失为每图求和的重建 BCE + β×按潜变量维求和的 KL。β=0 仍是带采样的编码器，不是确定性自编码器。

只使用 CPU。不要把 `CUDA_VISIBLE_DEVICES` 指到真实 GPU。不要改已有科研虚拟环境。

当前 `run_p1.py` 已支持 `--epochs`（默认 100）和 `--run-dir`。非默认 epoch 数必须同时给出 `--run-dir`，以免覆盖 100 epoch 基线。

**两次已完成的训练都发生在尚未提交的工作区上，当时 HEAD 是初始提交 `36ffabc`。本仓库后来的本地整理提交只用于保存可复现代码与索引，不能写成“该提交训练出了这些结果”。** 训练时源码快照与结果哈希见 `experiments/`。

## 目录

| 路径 | 说明 |
|---|---|
| `data_prep.py` | 二值化与 60/20/20 分层划分（seed=42） |
| `vae_model.py` | 模型、重参数化、损失，以及三项正确性测试入口 |
| `run_p1.py` | 训练 / 评估 / 出图；默认 100 epoch |
| `data/*_idx.npy` | 已固定的 train/val/test 索引（1078/359/360） |
| `data/x_all.npy`, `data/y_all.npy` | 全量二值化数据，保留在磁盘，不进 Git；可用同一脚本按相同种子再生 |
| `models/` | 100 epoch 末轮权重，保留在磁盘，不进 Git |
| `results/` | 100 epoch 指标、16 张图与潜变量，重文件不进 Git |
| `experiments/runs100/` | 100 epoch 运行清单（原路径 + sha256，不复制大文件） |
| `experiments/runs500/` | 500 epoch 运行清单与轻量指标副本；原件在 scratch 目录 |
| `experiments/source_snapshots/run_p1.py.100epoch` | 100 epoch 训练时使用的 `run_p1.py` 快照（当时还没有 `--epochs` / `--run-dir`） |
| `ENVIRONMENT.md` | 实际运行版本记录 |
| `requirements.txt` | 与上述记录对应的依赖清单，不是旧草稿钉 |

500 epoch 对照的原件在：

`/vepfs/wangshilong/scratch/beta-vae-digits-500epoch`

该目录不覆盖本项目的 `models/` 与 `results/`。

## 三项正确性测试

在已安装依赖的环境中：

```bash
cd /path/to/beta-vae-digits
CUDA_VISIBLE_DEVICES="" python vae_model.py
```

在原实验机上也可以用项目解释器：

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python vae_model.py
```

应看到：μ=0 且 logσ²=0 时 KL=0；合法高斯参数下 KL≥0；只对重建损失反传时，梯度能经重参数化到达 Encoder。已有通过记录：`results/metrics/correctness_tests.txt`。整理阶段的复跑日志写在 `experiments/verification/`，不会改写这份原始记录。

轻量入口检查：

```bash
CUDA_VISIBLE_DEVICES="" python run_p1.py --help
```

## 不要随意重跑的命令

已有结果必须保持原样。下面这些命令会覆盖 checkpoint、指标或图，**交付与归档时不要运行**：

```bash
python data_prep.py                  # 会重写 data/ 下的索引与数组
python run_p1.py all                 # 默认 100 epoch，写回本目录 models/ 与 results/
python run_p1.py eval-only --wall …  # 不训练，但会覆盖 results/ 中的评估输出与图
python run_p1.py all --epochs 500 --run-dir /path/to/run
```

若只是阅读已有结果，不必运行训练或 `eval-only`。若将来确需复现 500 epoch，必须给出独立的 `--run-dir`，并设 `CUDA_VISIBLE_DEVICES=` 与 `P1_NUM_THREADS=8`。

历史训练命令（仅作记录）见 `experiments/runs100/MANIFEST.md` 与 `experiments/runs500/MANIFEST.md`。

## 图怎么读

重建图是 μ 解码后的 sigmoid 概率；先验图是四组共用的 z ~ N(0, I) 解码；散点是全部测试 μ，类别只用于着色；插值是两个共用端点的 μ 线性插值后再解码。这些都不是 Bernoulli 采样，μ 解码也不是测试表里的 Reconstruction NLL。

报告指标是每图重建 NLL、KL 和标准负 ELBO（重建 NLL+KL）。不能直接比较不同 β 下的加权训练总损失。
