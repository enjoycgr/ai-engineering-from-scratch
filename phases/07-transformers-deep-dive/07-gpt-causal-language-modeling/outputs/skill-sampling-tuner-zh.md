---
name: sampling-tuner
description: 为给定生成任务选择解码策略（greedy / temperature / top-k / top-p / min-p / speculative）。
version: 1.0.0
phase: 7
lesson: 7
tags: [gpt, sampling, decoding, inference]
---

给定一个生成任务（代码、创意写作、推理、对话、结构化输出）以及延迟/质量目标，输出：

1. **采样方法。** 以下之一：greedy、temperature-only、top-k、top-p、min-p、beam-k、speculative。一句话说明理由。
2. **参数值。** Temperature、top-k、top-p、min-p、repetition penalty——与任务类型绑定的具体数值。（例如：temperature 0.2 + top-p 1.0 用于代码；min-p 0.1 + temperature 0.7 用于对话。）
3. **停止条件。** `max_new_tokens`、停止 token 列表、基于模式的停止（例如闭合的 `</tool_call>`）。
4. **确定性开关。** 固定种子用于可复现性；标记该用例（评估、法律场景）是否需要确定性输出。
5. **质量检查。** 针对任务目标的一行测试（编译/通过单元测试、事实性、格式有效性等）。

拒绝为结构化输出或代码补全推荐 temperature > 1.0——幻觉风险急剧上升。拒绝为开放式对话推荐纯 greedy——模型会陷入循环。拒绝在未指定停止 token 列表的情况下发布采样配置，尤其是当模型可能生成模板/工具调用时。
