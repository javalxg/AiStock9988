# AiStock9988

企业化、可审计、严格 PIT 的 A 股选股平台。旧项目只作为研究参考，不是运行时依赖。

当前阶段只完成架构与实验生命周期骨架。首个预注册实验为 `q70_source_parity_rebuild`：它使用 q70 的公开配置契约（F0=123、12 个月月度重训、周频、`rank:pairwise`、T+10、市场宽度配置），但不会复制旧模型、旧 ledger 或旧代码。

基础数据设施已包含只读 `SQLLoader`、CSV/Parquet `FileLoader`、稳定排序和 PIT 可见性过滤；集成测试使用临时 SQLite，不连接生产数据库。

## 初始化实验包

```bash
cd /Users/lxg/quant/AiStock9988
PYTHONPATH=src python3 -m aistock9988.cli init-run q70_source_parity_rebuild
```

正式运行要求 Git 工作区干净；每次运行拥有唯一 UTC ID，并先进入 `experiments/.running`。模型训练、全量预测账本、选择账本、订单/成交/NAV、数据 manifest 和审计完成后，必须执行：

```bash
PYTHONPATH=src python3 -m aistock9988.cli verify-run experiments/.running/<run_id>
PYTHONPATH=src python3 -m aistock9988.cli complete-run experiments/.running/<run_id>
```

完整训练、执行、止损和 NAV 回放入口：

```bash
PYTHONPATH=src python3 -m aistock9988.cli init-run q70_source_parity_rebuild
PYTHONPATH=src python3 scripts/first_q70_experiment.py --run-dir experiments/.running/<run_id>
PYTHONPATH=src python3 -m aistock9988.cli verify-run experiments/.running/<run_id>
PYTHONPATH=src python3 -m aistock9988.cli complete-run experiments/.running/<run_id>
```

审计未通过时不能完成或移动 run；系统不会用缺失产物生成“成功”结果。

## 首个正式实验口径

首个实验使用独立配置 `configs/experiments/q70_source_parity_t10_20260822.yaml`，实验名称为 `q70_source_parity_t10_365p63_mature`。正式统计边界为 `2026-07-31` 的完整成熟数据；历史 `2026-08-14` 的 `+386.02%` 只作为终端参考，不得混入正式结果。该实验固定 F0=123、T+10 成熟标签、月度重训、周频 Top2，并禁止读取旧 ledger、Stage2 和分钟数据。
