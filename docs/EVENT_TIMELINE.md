# Event Timeline v1

日线默认时序：

```text
T 收盘
→ 只读取 available_time <= T 收盘的数据
→ 生成特征与 MarketContext
→ 预测并冻结 Prediction/Selection Ledger
→ T+1 开盘检查停牌、涨跌停和可交易性
→ 使用 raw_open 成交
```

收盘发现的退出信号最早在 T+1 开盘成交。使用日内最低价推断成交必须标记为 `idealized_intraday_fill`，不可与普通日线成交混算。
