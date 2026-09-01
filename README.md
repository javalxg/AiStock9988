# AiStock9988

企业化、可审计、严格 PIT 的 A 股选股与回测平台。`deltafstation` 仅作只读研究参考，不是运行时依赖；迁移能力必须在本项目中重新设计数据契约、接口和审计。

## 当前主线

当前历史最优配置是 `reset_weak_confirm_v3_cap1_20`：弱市深跌后出现右侧收复时选股，T+1 原始开盘价成交，H10，前一收盘触发的 `-8%` 移动止损，单票使用决策 NAV 的 20%，最多五只，ADV20 参与率上限 2%。

2026-01-01 至数据库截止日 2026-08-28 的已见历史诊断：

| 场景 | 累计收益 | PF | MaxDD | 去最佳周 | 去前三赢家 | 交易数 | 胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | +32.44% | 2.254 | -8.27% | +20.12% | +16.23% | 41 | 56.1% |
| Stress | +28.48% | 2.058 | -8.82% | +16.57% | +12.86% | 41 | 56.1% |

完整决策见 [`RESET_WEAK_CONFIRM_V3_CAP1_20_FINAL_DECISION_20260901.md`](docs/council_20260828/RESET_WEAK_CONFIRM_V3_CAP1_20_FINAL_DECISION_20260901.md)。这些数字是历史诊断，不是未来收益承诺；33 周中只有 3 周达到 `+5%`，胜率也未达到 70%，因此用户目标仍未完成。

## 正式前向

CAP1 从 2026-09-01 起进入 append-only 前向锁箱。首日数据完整，但市场 20 日收益为 `+6.0043%`，冻结规则主动空仓，没有回退到随机买入或放松条件，也没有可报告收益。首日证据见 [`CAP1_EARLY_PATH_FORWARD_DAY1_FREEZE_20260901.md`](docs/council_20260828/CAP1_EARLY_PATH_FORWARD_DAY1_FREEZE_20260901.md)。

配置：

- 历史诊断：`configs/strategy/reset_weak_confirm_v3_cap1_20.yaml`
- 正式前向：`configs/strategy/reset_weak_confirm_v3_cap1_20_forward.yaml`
- 已预注册 shadow：`configs/strategy/cap1_early_path_forward_overlay.yaml`

运行时通过环境变量提供 QuantDB 凭据，密码不得写入配置、命令记录或实验产物。每个新完成交易日先执行预检：

```bash
cd /Users/lxg/quant/AiStock9988
PYTHONPATH=src python3 scripts/quiet_forward_preflight.py --asof YYYY-MM-DD
```

仅当预检返回 `READY_TO_FREEZE` 且交易日已经收盘，才能冻结当天选择：

```bash
PYTHONPATH=src python3 scripts/quiet_forward_shadow_runner.py \
  --mode freeze \
  --asof YYYY-MM-DD
```

执行数据成熟后，按同一信号日结算；不得提前补造未来价格：

```bash
PYTHONPATH=src python3 scripts/quiet_forward_shadow_runner.py \
  --mode settle \
  --asof YYYY-MM-DD \
  --execution-end YYYY-MM-DD
```

达到完整区间后生成汇总：

```bash
PYTHONPATH=src python3 scripts/quiet_forward_rollup_runner.py \
  --asof-start 2026-09-01 \
  --asof-end YYYY-MM-DD \
  --execution-end YYYY-MM-DD \
  --output docs/council_20260828/CAP1_FORWARD_ROLLUP
```

## 实验纪律

- 回测和冻结只使用 `trade_date <=` QuantDB 已有共同截止日的数据。
- 必需数据缺失只排除对应股票日；不得跳过整个交易日或随机补票。
- 新策略、模型、阈值、持仓、止损、排序或资金配置必须重新预注册并独立复核；不得用已见结果继续扫描。
- 正式报告必须同时给出 Base/Stress、PF、MaxDD、去最佳周、去前三赢家、周收益 `>=5%` 比例、胜率和最大持仓数。
- 审计未通过不得生成成功结果；锁箱只允许追加，禁止覆盖历史批次。
