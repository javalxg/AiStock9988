# deltafstation 能力迁移清单

| 旧项目能力 | 新项目目标 | 状态 | 迁移要求 |
|---|---|---|---|
| `backend/core/alpha/ml/sector_features.py` | `src/aistock9988/features/sector_relative.py` + `src/aistock9988/data/industry_pit.py` | PIT 解析和数据接线已实现，生产库待验收 | 使用 `index_member_all_ts` 的信号日 PIT 行业映射；禁止读取当前 `stock_basic_ts.industry` |
| `backend/core/alpha/ml/stock_xgb_walkforward.py` | `src/aistock9988/models/` | 部分实现 | 重新实现，不复制旧 runner；绑定冻结 F0 和标签契约 |
| q70 的 F0 日截面预处理（仅迁移能力，不迁移代码） | `src/aistock9988/features/f0_cross_section.py` | 实现中 | 123 列保持冻结顺序；每日独立 percentile/z-score；至少半数特征有效；训练截面确定性限 1500 行；禁止跨日填充 |
| q70 二阶段条件选择思想（仅迁移研究问题，不迁移模型或代码） | `scripts/causal_top20_event_selector_runner.py` | 已实现并否决该分类器 | Stage-1 历史 Top20 必须逐月因果生成；Stage-2 只在该 Top20 内训练与排序；标签成熟后方可训练；禁止阈值扫描、候选外扩和旧模型复用 |
| 旧数据加载器 | `src/aistock9988/data/` | 已有骨架 | 只读连接、查询哈希、PIT 审计、快照清单 |
| 历史 q70 source-parity 高收益配置（生产化参考契约） | `configs/experiments/q70_source_parity_t10_20260822.yaml` | 已登记，待专用 runner 验证 | 以 +365.63% 完整成熟边界为参考；+386.02% 仅历史终端参考；禁止旧 ledger、Stage2；使用 5 分钟生产级执行 |
| `top_list_ts` / `top_inst_ts` 龙虎榜事件数据契约（不迁移旧策略代码） | 独立的龙虎榜事件加载、去重、状态机和 V3 回测接线 | 设计审查中，未实现 | 只聚合 `exalter='机构专用'` 的 `net_buy`；`top_list` 原因保留集合且禁止跨原因累加 `net_amount`；事件截止日独立绑定；直接 T+1 追榜已否决，只允许预注册的回落-承接机制 |

## 行业相对特征迁移规则

旧实现按当前行业字段分组，不能直接迁移。新实现必须：

1. 对每个信号日 `T` 查询 `in_date <= T < out_date` 的有效行业关系；
2. 重叠关系按最新 `in_date`，再按最小 `index_code` 稳定解析；
3. 记录覆盖率和冲突数；
4. 行业映射快照纳入实验 `data_manifest.json`；
5. 完成 PIT 单元测试后，才能接入 q70 训练或回测。
