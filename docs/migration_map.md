# deltafstation 能力迁移清单

| 旧项目能力 | 新项目目标 | 状态 | 迁移要求 |
|---|---|---|---|
| `backend/core/alpha/ml/sector_features.py` | `src/aistock9988/features/sector_relative.py` | 计算逻辑已迁移，数据接线待完成 | 使用 `index_member_all_ts` 的信号日 PIT 行业映射；禁止读取当前 `stock_basic_ts.industry` |
| `backend/core/alpha/ml/stock_xgb_walkforward.py` | `src/aistock9988/models/` | 部分实现 | 重新实现，不复制旧 runner；绑定冻结 F0 和标签契约 |
| 旧数据加载器 | `src/aistock9988/data/` | 已有骨架 | 只读连接、查询哈希、PIT 审计、快照清单 |

## 行业相对特征迁移规则

旧实现按当前行业字段分组，不能直接迁移。新实现必须：

1. 对每个信号日 `T` 查询 `in_date <= T < out_date` 的有效行业关系；
2. 重叠关系按最新 `in_date`，再按最小 `index_code` 稳定解析；
3. 记录覆盖率和冲突数；
4. 行业映射快照纳入实验 `data_manifest.json`；
5. 完成 PIT 单元测试后，才能接入 q70 训练或回测。
