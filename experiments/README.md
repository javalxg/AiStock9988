# Experiment lifecycle

正式运行必须由项目 CLI 创建唯一 UTC run id，并先写入 `.running/<run_id>/`。只有完成配置/代码/数据哈希、PIT 审计、账本校验和结果写入后，才允许原子移动到 `completed/<run_id>/`。中断或失败移动到 `failed/<run_id>/`，不得覆盖既有目录。

386 复现的首个实验名称预注册为 `q70_source_parity_rebuild`，但新项目只复现配置契约，不复制旧模型或旧 ledger。
