# 本次实验的实际运行环境

以下版本来自已交付的两次 CPU 运行（见 `results/config.json` 与 `experiments/runs500/metrics/config.json`），不是未经验证的全新安装锁。仓库早期曾出现 `torch==2.0.1`、`numpy==1.24.3` 等草稿钉，那些**不是**本次实验使用的版本。

- 设备：CPU only（运行时 `CUDA_VISIBLE_DEVICES` 为空）
- Python：CPython 3.11.9
- 解释器（实验机）：`/vepfs/wangshilong/code/beta-vae-digits/.venv/bin/python`
- torch：2.7.1+cu126（从已有目录复用，在 CPU 上执行；没有把 torch 重装进本项目 site-packages）
- numpy：2.4.6
- matplotlib：3.11.2
- scipy：1.17.1
- scikit-learn：1.9.1
- 另外安装到项目 venv 的包：joblib、threadpoolctl、cloudpickle、narwhals
- 复用的 site-packages：`/vepfs/wangshilong/envs/worldmodel-wan/lib/python3.11/site-packages`
- 未改动科研虚拟环境 `.venv-wmr`、`.venv-state-trajectory`

## 两次已完成运行的共同设置

β ∈ {0, 0.1, 1, 4}；Adam 学习率 1e-3；batch size 128，不 drop_last；训练 1078 / 验证 359 / 测试 360；训练种子 42；划分种子 42；每 epoch shuffle 种子为 `42+epoch`；评估种子 12345；每图 20 次后验 MC；每 epoch 9 次更新（8×128 + 最后一批 54）。交付的是该预算的**末轮**模型，验证集只做监控，不用测试集选模。

- 100 epoch 基线：每 β 900 次 `optimizer.step`；四组并行墙钟 11.399 秒；结果在本仓库的 `models/` 与 `results/`。
- 500 epoch 对照：每 β 4500 次 `optimizer.step`；四组并行墙钟 33.725 秒；结果在 `/vepfs/wangshilong/scratch/beta-vae-digits-500epoch`，不覆盖基线。

路径、sha256、命令与训练时源码快照见 `experiments/`。本目录不附带 `.venv` 或缓存。在其他机器上复现测试时，请自行准备兼容的 CPU 版依赖，并以本文件中的版本为参考；未在本仓库内验证“从零 pip 安装上述精确构建”的一条命令。
