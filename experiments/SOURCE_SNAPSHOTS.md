# 训练时源码快照（诚实映射）

两次实验运行时，仓库 HEAD 都是初始提交：

`36ffabc747b720b48aca25ff44039b1244a5bf5a`

该提交只包含 `.gitignore`，还没有 `data_prep.py` / `vae_model.py` / `run_p1.py`。因此**不存在“训练所用的 Git commit”**。后面为整理代码而做的本地提交，只用于保存可复现入口和索引，**不能倒称它训练出了这些 checkpoint**。

## 两次运行都未改动的文件

| 文件 | 字节 | sha256 |
|---|---:|---|
| `data_prep.py` | 1352 | `a212418acc4c1b82c6ab9c22df7bf382fee473e52147af24c728d6e1d70a956c` |
| `vae_model.py` | 4140 | `f7071600f89e9ca7f1a4dc96de03131be4c4026fb3df57bf663f56d082928a33` |

## `run_p1.py` 在两次运行之间有变化

| 运行 | 字节 | sha256 | 说明 |
|---|---:|---|---|
| 100 epoch | 30935 | `082f2c4f5282a3e9480c1084d6e4d623bfd508bdbf74fa439ea833297377ec02` | 当时还没有 `--epochs` / `--run-dir`。完整快照：`experiments/source_snapshots/run_p1.py.100epoch` |
| 500 epoch | 44228 | `158c0802ab6c142eaaed25b2ecef99a11fdd4a0f962eee8cf9ea283bbd0cfaf8` | 当前项目中的 `run_p1.py`。默认仍是 100 epoch；非默认必须带 `--run-dir`。500 epoch 结束时它仍未提交 |

整理提交如果没有改这三个实现文件，则当前 `data_prep.py` / `vae_model.py` / `run_p1.py` 的哈希应继续与上表中的 500 epoch 行（以及共用的前两行）一致。
