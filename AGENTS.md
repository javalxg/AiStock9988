# AiStock9988 工作边界

## 主项目

当前唯一主开发目录是：`/Users/lxg/quant/AiStock9988`。

所有新代码、测试、配置、实验和文档必须写入本目录。默认不得修改或依赖 `/Users/lxg/quant/deltafstation` 的运行时代码、模型、缓存、ledger 或数据库访问实现。

## deltafstation 的定位

`deltafstation` 仅作为只读研究资料和历史结论参考。迁移能力时必须重新设计接口、数据契约和 PIT 检查，不能直接复制旧文件。

## 实验边界

- 先修复数据时间、标签成熟度和预测 PIT，再启动正式训练。
- MySQL 只读访问必须通过 `aistock9988.data.quantdb`。
- 每次实验使用独立目录并记录配置、命令、代码哈希、数据快照和结果。
- 任何行业、因子或执行逻辑从旧项目迁移前，先在 `docs/migration_map.md` 登记来源、目标模块和验证测试。
