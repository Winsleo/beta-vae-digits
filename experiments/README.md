# 实验索引

这里只保存可进 Git 的轻量清单、少量指标副本，以及训练时的源码快照说明。大文件仍留在原来的磁盘路径，用 sha256 对照，不在这里复制一份 checkpoint 或全套图。

| 清单 | 原件位置 | 已交付快照（只读，勿改） |
|---|---|---|
| `runs100/` | `/vepfs/wangshilong/code/beta-vae-digits/{data,models,results}` | `/workspace/exports/p1-beta-vae-final/` 与 `p1-beta-vae-final.zip` |
| `runs500/` | `/vepfs/wangshilong/scratch/beta-vae-digits-500epoch` | `/workspace/exports/p1-beta-vae-500epoch/` 与 `p1-beta-vae-500epoch.zip` |

两次运行的共同协议：划分 1078/359/360，β ∈ {0, 0.1, 1, 4}，每 epoch 9 次更新，评估 20 次 MC、种子 12345，展示索引与先验 z 共用。100 epoch = 每 β 900 步；500 epoch = 每 β 4500 步。

**训练时没有对应的功能提交。** 当时 HEAD 都是 `36ffabc`。详见 `SOURCE_SNAPSHOTS.md`。不要把后来的整理提交说成训练提交。

`runs500/COMPARISON.volc.md` 是火山原件的轻量副本。box 已交付包里的 `COMPARISON.md` 是打包时改写过的版本，哈希不同；已交付 zip 不要改，也不要用它覆盖火山原件。
