# AiStock9988

企业化、可审计、严格 PIT 的 A 股选股平台。旧项目只作为研究参考，不是运行时依赖。

当前阶段只完成架构与实验生命周期骨架。首个预注册实验为 `q70_source_parity_rebuild`：它使用 q70 的公开配置契约（F0=123、12 个月月度重训、周频、`rank:pairwise`、T+10、市场宽度配置），但不会复制旧模型、旧 ledger 或旧代码。

基础数据设施已包含只读 `SQLLoader`、CSV/Parquet `FileLoader`、稳定排序和 PIT 可见性过滤；集成测试使用临时 SQLite，不连接生产数据库。

## 初始化实验包

```bash
cd /Users/lxg/quant/AiStock9988
PYTHONPATH=src python -m aistock9988.cli init-run q70_source_parity_rebuild
```

正式运行要求 Git 工作区干净；每次运行拥有唯一 UTC ID，并先进入 `experiments/.running`。模型训练、全量预测账本、选择账本、订单/成交/NAV 和审计完成后，才允许标记 `COMPLETED` 并移动到 `experiments/completed`。
