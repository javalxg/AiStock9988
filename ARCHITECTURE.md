# AiStock9988 量化选股平台架构（设计冻结稿 v1）

## 0. 设计原则

AiStock9988 是一个全新的、可审计的企业化项目。旧 `deltafstation` 只作为研究资料读取，禁止直接复制运行时代码、模型、缓存或 ledger。

核心原则：

1. **数据、训练、预测、组合、执行、报告分层**，任何层不能越权读取未来数据。
2. **每次运行是不可变实验包**：代码、配置、数据快照标识、模型、日志、结果、哈希在同一目录。
3. **回测和实盘选股共享同一 Selection Contract**，差别只在数据时间点和执行适配器。
4. **市场宽度是一级输入**，与指数趋势、行业广度、资金状态共同组成 `MarketContext`，不能事后用收益过滤。
5. **模型只负责预测和排序，风险/交易规则独立**；Stage2 不是默认存在的模块。
6. **所有扩展采用 SPI 注册机制**，新增因子或过滤器只能注册新插件，不修改主流程。

## 1. 目录结构（先建空骨架，后实现）

```text
AiStock9988/
├── ARCHITECTURE.md
├── pyproject.toml
├── configs/                         # 只放版本化、可审阅配置
│   ├── data_contract.yaml
│   ├── feature_sets/
│   ├── model_profiles/
│   └── execution_profiles/
├── src/aistock9988/
│   ├── domain/                      # 纯领域对象，无 DB/模型依赖
│   ├── data/                        # 数据适配器、PIT 查询、快照
│   ├── features/                    # 因子族与特征快照
│   ├── market/                      # 指数、宽度、行业、资金状态
│   ├── labeling/                    # 可配置标签与成熟检查
│   ├── models/                      # 训练、预测、模型注册
│   ├── selection/                   # Top20→最终组合，默认无 Stage2
│   ├── execution/                  # 复权价格执行 + 原始可交易状态
│   ├── backtest/                    # 事件驱动回测
│   ├── reporting/                   # 周收益、胜率、回撤、诊断
│   └── spi/                         # 少量稳定边界的显式插件注册表
├── baselines/                       # B0/B1/B2 晋级记录
├── tests/                           # unit/integration/golden/leakage/accounting
├── experiments/
│   ├── .running/                    # 运行中，不得用于比较
│   ├── completed/                   # 仅允许 COMPLETED
│   └── failed/
│       └── <UTC>_<name>_<config8>/  # 每次训练/回测一个独立实验包
│       ├── config.yaml
│       ├── code_manifest.json
│       ├── data_manifest.json
│       ├── commands.sh
│       ├── models/
│       ├── predictions/
│       ├── selections/
│       ├── trades/
│       ├── diagnostics/
│       ├── logs/
│       └── RESULT.md
└── docs/                            # 设计决策、数据字典、验收报告
```

## 2. 运行边界

```text
数据源(DB/文件)
      │  PIT 查询 + 质量元数据
      ▼
DataSnapshot ──► FeatureSnapshot ──► MarketContext
      │                │                  │
      └──────────────► TrainingDataset ◄──┘
                              │ 只用已成熟标签
                              ▼
                     WalkForwardTrainer
                              │ ModelArtifact
                              ▼
                 Predictor → PredictionLedger(全量)
                              │ 冻结视图
                              ▼
                        CandidateLedger(Top20)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       SelectionPolicy                   Audit/Report
       (默认直接排序)                     (不改选择)
               │
               ▼
       ExecutionAdapter → OrderLedger → FillLedger → Position/NAV Ledger
```

## 3. q70 基线契约（仅作为配置，不复制旧实现）

第一版 baseline 只重建已经明确的契约：

- F0=123：57 技术 + 57 行业相对 + 9 基本面；因子列顺序写入 `feature_set_id` 和哈希。
- 月度重训、12 个月训练窗口、每周信号。
- `rank:pairwise`，seed=42，depth/树数/正则项全部显式写入配置。
- 信号日 T，T+1 开盘执行；预测期限由 label profile 配置，不能写死在平台中。
- 标签成熟必须满足 `label.available_time <= model_training_cutoff`；索引边界检查只是辅助校验。
- `economic_price` 用于连续收益和技术指标；订单/成交使用 `raw_price`；NAV 使用真实股数、现金和公司行动账本。
- Top20 是候选视图，不是唯一预测记录；必须同时保存全量 `PredictionLedger`。
- 最终持仓数量由 SelectionPolicy 配置，不固定写死 Top2。
- 止损、止盈、技术退出、仓位上限全部属于 execution profile，不进入训练特征。
- 市场上下文包括：指数多周期趋势、全市场上涨/下跌家数及比例、涨跌停/炸板、行业上涨宽度、成交额与波动状态、资金流状态。每个字段都必须有 PIT 时间戳和计算窗口。

注意：q70 的“最好收益数字”不是本项目的验收依据。第一验收依据是能否在同一数据快照、同一代码哈希、同一配置下稳定重跑并得到同一 ledger；收益指标随后才有意义。

标签 profile 示例：

```yaml
label:
  id: endpoint_open_open_v1
  entry_delay_sessions: 1
  horizon_sessions: 5       # 可改为 10，不属于平台常量
  entry_price: economic_open
  exit_price: economic_open
  maturity_sessions: 6
  clip: [-0.30, 0.30]
```

标签注册表至少支持 T+5、T+10、固定持有、路径收益、超额收益和模拟退出标签。

## 4. SPI 扩展协议

插件接口保持窄而稳定：

```text
FeatureProvider.build(snapshot, asof) -> FeatureBlock
MarketContextProvider.build(snapshot, asof) -> MarketContextBlock
SelectionPlugin.apply(candidates, context, asof) -> SelectionDecision
ExecutionRule.evaluate(position, market, asof) -> ExitDecision
ReportPlugin.render(run_artifact) -> ReportBlock
```

插件必须声明：名称、版本、输入字段、PIT 要求、输出字段、缺失处理、参数哈希。插件不能直接查询未来日期，不能修改其他插件输出，不能写共享缓存。

Stage2 仅在配置中显式启用时加载；启用后必须记录其输入仍来自冻结 Top20，且不得替换 baseline 的原始候选账本。

## 5. 数据与泄漏审计

每个 `DataSnapshot` 记录：表、查询条件、最大可见日期、行数、列哈希、数据库连接标识（不保存密码）。

每个模型头记录：训练开始/结束日期、成熟边界、样本数、特征签名、模型文件哈希、运行时版本。

每次预测前自动执行：

1. 特征最大日期不得晚于 T；
2. 训练标签退出日期不得晚于训练/预测时点；
3. 市场宽度和行业状态只能使用 T 收盘前数据；
4. 预测 ledger 先落盘，再计算任何未来收益诊断；
5. 诊断数据与交易决策数据物理分目录，防止事后字段回流。

## 6. 开发与验收顺序

1. 先实现领域对象、配置校验、快照/哈希和 PIT 审计（不接模型）。
2. 再实现数据适配器和市场上下文，先做全市场宽度单元测试。
3. 再实现标签成熟器与 walk-forward 训练器，使用小样本 smoke test。
4. 再实现 q70 baseline 的模型 profile 和预测 ledger。
5. 最后实现执行回放与报告；每周收益、胜率、PF、最大回撤、去最大盈利周均自动生成。
6. 2024、2025、2026-08 以前属于开发与稳健性评估区间；2026-08 以后为前向锁箱。每个实验同时报告开发区间与锁箱区间，禁止按年份挑选结论。

## 7. 实验命名与不可变要求

实验目录一经开始运行不得覆盖。修正任何代码、配置、数据查询或依赖都必须新建目录。`commands.sh` 是唯一官方启动入口，禁止用户直接调用内部模块。

每个实验完成后必须写 `RESULT.md`：目标、实际命令、哈希、数据范围、关键指标、失败周、结论、是否进入 baseline。无结果的运行不得口头宣称有效。

## 8. 外围能力边界

研究、每日复盘、个股诊断、持仓管理和日内做 T 共享数据快照与领域对象，但不允许反向改变选股结果。复盘结论、人工标注和诊断输出默认只进入 `review/diagnosis`，经独立实验晋升后才能成为特征。

```text
data/ → selection/backtest
     → review
     → diagnosis
     → portfolio → intraday
```

这些模块采用模块化单体，暂不拆微服务；实时行情采集可作为独立后台任务。波浪和未收盘缠论结果只能作为诊断展示，必须带 `provisional` 标记，不能直接成为自动交易条件。
