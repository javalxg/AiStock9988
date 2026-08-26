# Accounting Contract v1

研究收益与真实成交严格分离。`raw_price` 是订单和成交价格，`economic_price` 只用于连续收益/技术指标；公司行动由股数与现金账本处理。

```text
cash_after_buy  = cash_before_buy - raw_fill_price*shares - buy_commission
cash_after_sell = cash_before_sell + raw_fill_price*shares - sell_commission - stamp_duty
realized_pnl    = net_sell_proceeds - allocated_cost_basis
nav             = cash + Σ(shares * raw_mark_price)
PF              = Σ(realized_pnl>0) / abs(Σ(realized_pnl<0))
```

报告必须分别列出 `trade_win_rate`、`equal_trade_return_ratio` 和 `portfolio_profit_factor`。每次运行执行 NAV 恒等式、现金流守恒、持仓股数守恒测试。
