# runs500：500 epoch 对照索引

原件（不要改）：`/vepfs/wangshilong/scratch/beta-vae-digits-500epoch`

已交付只读快照：`/workspace/exports/p1-beta-vae-500epoch/`  
已交付 zip（勿改）：`/workspace/exports/p1-beta-vae-500epoch.zip`  
sha256：`754169992f37dcdc71c71c213e368fad3b2fba7dcfba9f3c431685b82ee0c0db`

该 zip 中的模型、指标、图和 epoch100 快照与火山原件一致。zip 内 `COMPARISON.md` 是打包时改写版（sha256 `55c8854dde16bd0dc2626cdb4d729966fa90c067a73351ccdc3a548740e1e339`），与火山原件 `9a567c08e08be1e895c85c89d436fe5ef73fbddfa5983a85b6c1e870896345ae` 不同。Git 里保存的是火山原件副本 `COMPARISON.volc.md`。不要用打包版覆盖原件，也不要改已交 zip。

轻量指标副本在 `metrics/`。checkpoint 与全套图只记录原路径和哈希。

## 协议与命令

- 与 100 epoch 同一划分、多种子约定和评估协议；从种子 42 **从头重训**到 500，不是热启动
- 每 β **4500** 步；并行墙钟 33.725 秒；`P1_NUM_THREADS=8` 未重新选线程
- 历史命令（不要为归档再跑）：`CUDA_VISIBLE_DEVICES="" P1_NUM_THREADS=8 .venv/bin/python run_p1.py all --epochs 500 --run-dir /vepfs/wangshilong/scratch/beta-vae-digits-500epoch`

## 训练时代码

HEAD 仍是 `36ffabc`，实现仍未提交。当时的 `run_p1.py` 已是当前这份（`--epochs` / `--run-dir`），哈希 `158c0802ab6c142eaaed25b2ecef99a11fdd4a0f962eee8cf9ea283bbd0cfaf8`（44228 字节）。

## 测试指标

每 β 训练步数：**4500**。测试集 360 张，20 次 MC，seed=12345。

| β | Recon NLL | KL | 标准负 ELBO |
|---|---:|---:|---:|
| 0 | 16.9926 | 97.6064 | 114.5990 |
| 0.1 | 16.3120 | 7.6766 | 23.9886 |
| 1 | 16.9103 | 3.2577 | 20.1680 |
| 4 | 23.4790 | 0.4105 | 23.8895 |

epoch 100 快照的测试指标、前 100 轮 history 和权重量差相对 100 epoch 基线均为 0。明细见 `metrics/reproduction_check.json`。
