# Data Time Model v1

每条数据统一记录三种时间：

- `event_time`：数据描述的市场时点；
- `available_time`：策略最早能够看到的时间；
- `ingested_time`：写入本地数据源的时间。

PIT 判断统一使用 `available_time <= decision_time`。快照还要记录 source id、查询哈希、分区 checksum、行数、schema/content hash、可见日期范围和提取时间。数据库密码不得写入 manifest。
