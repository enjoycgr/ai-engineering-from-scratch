---
name: document-ai-stack-picker
description: 基于领域、规模和监管需求为文档 AI 项目在 OCR 流水线、OCR-free 专家和 VLM-native 之间做选择。
version: 1.0.0
phase: 12
lesson: 22
tags: [document-ai, ocr, donut, nougat, paligemma, vlm-native]
---

给定文档 AI 项目（领域：发票/科学论文/表单/混合；规模：每天页数；质量标准；监管需求），选择栈并产出参考配置。

产出：

1. 栈选择。时代 1（OCR 流水线 + LayoutLMv3）、时代 2（Donut / Nougat OCR-free）、时代 3（VLM-native）、或混合。
2. 每页成本估算。所选栈的 token 计数和延迟。
3. 准确率预期。DocVQA + ChartQA + 领域特定基准。
4. 手写策略。成本不敏感用 VLM-native；规模用专用 TrOCR + 路由。
5. 数学 / LaTeX 输出。科学论文用 Nougat；其他用 VLM。
6. 监管回退。带交叉检查审计日志的混合。

硬性拒绝：
- 未经成本分析就为 >100 万页/天提议 VLM-native。2576px 每页的 token 成本显著。
- 为监管工作流推荐无审计路径的单一模型方案。
- 声称 Nougat 处理扫描发票。它不能 — 它是科学论文专家。

拒绝规则：
- 如果规模 >1000 万页/天，拒绝时代 3 并推荐时代 1 以时代 3 作为采样验证器。
- 如果领域是手写密集，拒绝 OCR 流水线并推荐 VLM-native + 手写专家（TrOCR）。
- 如果方程需要 LaTeX 保真度，要求在循环中加入 Nougat。

输出：一页计划，含栈、成本、准确率、手写、数学、监管。结尾附 arXiv 2308.13418（Nougat）、2204.08387（LayoutLMv3）、2111.15664（Donut）。
