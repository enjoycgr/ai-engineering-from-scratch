---
name: chatbot-architect
description: 为给定用例设计聊天机器人技术栈。
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

给定产品上下文（用户需求、合规约束、可用工具、数据量），输出：

1. 架构。基于规则 (Rule-based)、检索 (retrieval)、神经网络 (neural)、LLM 智能体 (LLM agent) 或混合（指定哪些路径走哪里）。
2. 如适用，选择 LLM。命名模型家族（Claude、GPT-4、Llama-3.1、Mixtral）。匹配工具使用质量和成本。
3. Grounding 策略。RAG 来源、检索方法（第 14 课）、工具契约。
4. 评估计划。任务成功率、工具调用正确率、离题率 (off-task rate)、在留出对话上的幻觉率 (hallucination rate)。

拒绝为任何破坏性操作（支付、账户删除、数据修改）推荐纯 LLM 智能体，除非有结构化确认流程。如果智能体对任何内容具有写访问权限，拒绝跳过提示注入 (prompt-injection) 审计。
