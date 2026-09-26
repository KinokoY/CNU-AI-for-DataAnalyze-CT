# 整卷评估结果

本目录只收录可交付的评估结果，不收录原始影像、标签、缓存、预测卷或模型权重。`configs/default.yaml` 的 `paths.reports` 已指向本目录。

在远程平台的仓库根目录，完成本地代码的 `git pull` 后运行：

```bash
python -m src.evaluate --fold 0
```

这条命令默认读取 `runs/fold0/best.pt` 和远程 `cache/`，生成：

- `reports/eval_fold0.json`：fold 0 的逐例原始评估结果与折级汇总。
- `reports/eval_summary.json`：本次评估的结构化汇总；当前只有 fold 0 时，它仍是单折汇总。
- `reports/eval_summary.md`：供案例撰写与验收阅读的报告。

确认命令成功结束后，在远程平台只提交上述结果：

```bash
git add reports/eval_fold0.json reports/eval_summary.json reports/eval_summary.md
git commit -m "加入 fold 0 整卷评估结果"
git push
```

本地随后执行 `git pull`，即可同时取得代码与评估结果。若未来完成其他折，再提交对应的 `reports/eval_fold1.json` 至 `reports/eval_fold4.json`，并重新生成、提交两份 `eval_summary`。报告中的 `folds_evaluated` 才是实际完成的折数；只有 fold 0 时不要写成五折结果。

不要使用 `--save-pred` 作为交付命令。`runs/`、`cache/`、`data/` 中的受限数据仍按仓库规则忽略；`reports/` 也仅放行上述评估文件。
