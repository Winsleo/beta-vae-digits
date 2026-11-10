# runs100：100 epoch 基线索引

原件（不要改）：`/vepfs/wangshilong/code/beta-vae-digits/{data,models,results}`

已交付只读快照：`/workspace/exports/p1-beta-vae-final/`  
已交付 zip（勿改）：`/workspace/exports/p1-beta-vae-final.zip`  
sha256：`244ce9f09f6939ea633125c3e36ac5609dc7beb73d559316913992e204a3576a`

完整文件表（原路径 + sha256 + 是否进 Git）见同目录 `MANIFEST.json`。本清单不复制 checkpoint 或图。

## 协议与命令

- 划分 1078/359/360，seed=42；β ∈ {0, 0.1, 1, 4}；100 epoch 末轮；每 epoch 9 次更新，每 β **900** 步
- 评估 20 次 MC，seed=12345；图示索引与先验 z 见 `results/latent_codes/`
- 并行墙钟 11.399 秒；`P1_NUM_THREADS=8`，四进程
- 历史命令（不要为归档再跑）：`CUDA_VISIBLE_DEVICES="" P1_NUM_THREADS=8 .venv/bin/python run_p1.py all`

## 训练时代码

HEAD=`36ffabc`，实现文件尚未提交。`run_p1.py` 快照哈希 `082f2c4f5282a3e9480c1084d6e4d623bfd508bdbf74fa439ea833297377ec02`（30935 字节），见 `experiments/source_snapshots/run_p1.py.100epoch`。

## 测试指标

每 β 训练步数：**900**。测试集 360 张，20 次 MC，seed=12345。

| β | Recon NLL | KL | 标准负 ELBO |
|---|---:|---:|---:|
| 0 | 16.9989 | 48.2455 | 65.2444 |
| 0.1 | 17.3173 | 6.4404 | 23.7576 |
| 1 | 18.2536 | 2.6029 | 20.8564 |
| 4 | 24.2155 | 0.2527 | 24.4683 |

标准负 ELBO = Recon NLL + KL。四组模型 sha256 见 JSON；它们与 500 epoch 目录中 `snapshots/epoch100/` 的对应权重完全一致。
