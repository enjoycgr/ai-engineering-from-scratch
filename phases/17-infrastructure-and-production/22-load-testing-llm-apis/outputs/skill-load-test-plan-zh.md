---
name: load-test-plan
description: 设计真实的 LLM 负载测试 —— 选择工具（LLMPerf、k6、GenAI-Perf、guidellm），构建四种模式（steady、ramp、spike、soak），并在 CI 中设置门禁。
version: 1.0.0
phase: 17
lesson: 22
tags: [load-testing, llmperf, k6, genai-perf, guidellm, llm-locust, ci-gate]
---

给定工作负载（端点、TTFT/TPOT/错误的 SLA）、目标规模（并发、RPS）和 CI 策略（PR 门禁或仅发布时），产出负载测试计划。

产出：

1. 工具。基线运行用 LLMPerf；CI 门禁用 k6 + 流式扩展；NVIDIA 参考运行用 GenAI-Perf；大规模合成测试用 guidellm。仅当已有 Locust 时才用 LLM-Locust。
2. 提示词分布。从真实流量获取输入 token 的 mean + stddev（如有）或使用已发布分布（ShareGPT / HumanEval）。禁止 loop-with-one-prompt。
3. 四种模式。Steady、ramp、spike、soak。每种需包含：目标 RPS、持续时间、预期故障模式。
4. CI 门禁。具体阈值：TTFT P95 < X、5xx < 5%、TPOT < Y。每个 PR 运行时间：3-5 分钟。
5. 指标对齐。注意报告工具是 GenAI-Perf 风格（ITL 排除 TTFT）还是 LLMPerf 风格（ITL 包含 TTFT）。选择一种并保持统一。
6. 产出。提交到仓库的脚本文件（k6 JS、LLMPerf CLI）。

硬性拒绝：
- 使用统一提示词进行负载测试。拒绝 —— 数字会撒谎。
- 不支持流式传输的负载测试。拒绝 —— LLM 端点默认使用流式传输。
- 跨工具比较数字而不承认指标定义差异。拒绝。

拒绝规则：
- 如果团队打算使用未安装 LLM-Locust 扩展的 Locust 原版，拒绝 —— GIL 陷阱。
- 如果 CI 门禁预算 < 每个 PR 60 秒，拒绝完整 soak —— 建议快速 steady-state 加独立夜间 soak。
- 如果提示词分布数据不可用，要求使用已记录的已发布分布（ShareGPT）并注明假设。

产出：一页计划，包含工具、提示词分布、四种模式及目标、CI 门禁阈值、指标对齐。最后给出单一 CI 产出：仅当所有阈值达标且 3 次运行稳定时 PR 才通过。
