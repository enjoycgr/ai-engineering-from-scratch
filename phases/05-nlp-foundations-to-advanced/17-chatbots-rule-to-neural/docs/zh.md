# 聊天机器人 —— 从基于规则到神经网络再到 LLM 智能体

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中生成答案。Claude 调用工具并验证。每一个时代都解决了前一个时代最糟糕的失败。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 分钟

## The Problem

用户说 "I want to change my flight"。系统必须弄清楚他们想要什么、缺少什么信息、如何获取信息以及如何完成操作。然后用户说 "wait, what if I cancel instead?"，系统必须记住上下文、切换任务并保持状态。

对话对机器学习系统来说很困难。输入是开放式的。输出必须在多轮对话中保持连贯。系统可能需要对世界采取行动（改签航班、刷卡）。每一步错误对用户都是可见的。

聊天机器人架构经历了四种范式，每一种都是因为前一种失败得太明显而引入的。本课按顺序介绍它们。2026 年的生产环境是最后两种的混合体。

## The Concept

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**基于规则的 (Rule-based)（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器将请求路由到预定义的流程。槽填充 (slot-filling) 状态机收集所需信息。在其设计的狭窄范围内表现出色。一旦超出范围立即失败。仍然部署在安全关键领域（银行身份验证、航空公司预订），因为这些领域不容忍幻觉 (hallucination)。

**基于检索的 (Retrieval-based)。** 一种 FAQ 风格的系统。编码每一对（话语，回复）。在运行时，编码用户的消息并检索最接近的存储回复。想想 Zendesk 经典的 "similar articles" 功能。比规则更好地处理改写。不生成文本，因此没有幻觉。

**神经网络的 (Neural)（seq2seq）。** 在对话日志上训练的编码器-解码器模型。从头生成回复。流畅但容易产生通用输出（"I don't know"）和事实漂移。从未可靠地保持在主题上。这就是 Google、Facebook 和 Microsoft 在 2016-2019 年都有令人失望的聊天机器人的原因。

**LLM 智能体 (LLM agents)。** 一个被包装在循环中的语言模型，能够规划、调用工具并验证结果。不是一个带有长提示词的聊天机器人。而是一个智能体循环 (agent loop)：plan → call tool → observe result → decide next step。检索优先的 grounding（RAG）防止幻觉。工具调用让它真正做事。这就是 2026 年的架构。

这四种范式不是顺序替代关系。2026 年的生产聊天机器人会路由通过所有四种：基于规则的处理身份验证和破坏性操作，检索处理 FAQ，神经生成处理自然措辞，LLM 智能体处理模糊的开放式查询。

## Build It

### Step 1: 基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码实现 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年经典心理治疗师演示。至今仍具有启发性。

### Step 2: 基于检索的 (FAQ)

这个说明性代码片段需要 `pip install sentence-transformers`（会拉取 torch）。本课可运行的 `code/main.py` 使用标准库的 Jaccard 相似度，因此本课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键的设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### Step 3: 神经生成（基线）

使用小型指令微调的编码器-解码器模型（FLAN-T5）或微调的对话模型。2026 年单独使用它在生产上不可用（矛盾、离题漂移、事实胡说），但在混合系统中用于自然措辞。DialoGPT 风格的仅解码器模型需要显式的轮次分隔符和 EOS 处理才能生成连贯的回复；FLAN-T5 text2text pipeline 对教学示例来说开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Step 4: LLM 智能体循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

有三件事需要说明。Tools 是 LLM 可以调用的可调用函数。当 LLM 返回最终答案而不是工具调用时，循环终止。Step budget 防止模糊任务上的无限循环。

真正的生产环境还会添加：检索优先的 grounding（在每次 LLM 调用前注入相关文档）、guardrails（拒绝未经确认的破坏性操作）、observability（记录每一步）和 evaluations（自动检查智能体行为是否符合规范）。

### Step 5: 混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：对任何破坏性操作使用确定性规则，对固定 FAQ 使用检索，对其他所有内容使用 LLM 智能体。这就是 2026 年客户支持系统中实际部署的方式。

## Use It

2026 年的技术栈：

| Use case | Architecture |
|---------|---------------|
| Booking, payment, authentication | 基于规则的状态机 + 槽填充 |
| Customer support FAQs | 对精选答案的检索 |
| Open-ended help chat | 带 RAG + 工具调用的 LLM 智能体 |
| Internal tools / IDE assistants | 带工具调用的 LLM 智能体（搜索、读取、写入） |
| Companion / character chatbots | 带人设系统提示词的微调 LLM，基于知识的检索 |

生产中始终使用混合路由。没有单一架构能很好地处理每个请求。路由层本身通常是一个小型意图分类器。

## Failure modes that still ship

- **自信的虚构 (Confident fabrication)。** LLM 智能体声称完成了一个它并未完成的操作。缓解措施：验证结果、记录工具调用、永远不要让 LLM 声称在没有成功工具返回的情况下完成了某事。
- **提示注入 (Prompt injection)。** 用户插入覆盖系统提示词的文本。在 OWASP Top 10 for LLM Applications 2025 中排名第一（LLM01）。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编码基准测试中，前沿模型的测量成功率约为 0.5-8.5%。特定的高风险设置（针对 AI 编码智能体的自适应攻击、脆弱的编排）已达到约 84%。生产环境中的 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）—— 由攻击者控制的电子邮件触发的 Microsoft 365 Copilot 零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信；在工具调用前进行清理；将工具输出与主提示词隔离；使用 Plan-Verify-Execute (PVE) 模式，智能体先规划，然后针对该计划验证每个动作再执行（这阻止了工具结果注入新的未计划动作）；对破坏性操作需要用户确认；对工具范围应用最小权限原则。

  再多的提示工程也无法完全消除这种风险。需要外部运行时防御层（LLM Guard、allowlist 验证、语义异常检测）。
- **范围蔓延 (Scope creep)。** 智能体偏离任务，因为工具调用返回了间接相关的信息。缓解措施：缩小工具契约；保持系统提示词聚焦；添加对 off-task rate 的 evaluations。
- **无限循环 (Infinite loops)。** 智能体不断调用同一个工具。缓解措施：step budget、工具调用去重、LLM judge 判断 "are we making progress"。
- **上下文窗口耗尽 (Context window exhaustion)。** 长对话将最早的轮次推出上下文。缓解措施：总结较早的轮次、通过相似度检索相关的过去轮次，或使用长上下文模型。

## Ship It

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Exercises

1. **Easy.** 为咖啡店点餐机器人实现上述基于规则的回复，包含 10 个模式。测试边界情况：双份订单、修改、取消、意图不明确。
2. **Medium.** 构建一个混合 FAQ + LLM 降级系统。为 SaaS 产品准备 50 条固定 FAQ 条目，LLM 降级使用对文档站点的检索。在 100 个真实支持问题上测量拒绝率和准确率。
3. **Hard.** 用三个工具（搜索、读取用户数据、发送邮件）实现上述智能体循环。运行包含 50 个测试场景的评估，包括提示注入尝试。报告 off-task rate、failed task rate 和任何注入成功。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Intent | What the user wants | 分类标签（book_flight, reset_password）。路由到处理程序。 |
| Slot | A piece of info | 机器人需要的参数（date, destination）。Slot filling 是询问的序列。 |
| RAG | Retrieval plus generation | 检索相关文档，然后为 LLM 的回复提供依据。 |
| Tool call | Function invocation | LLM 发出带有 name + args 的结构化调用。运行时执行并返回结果。 |
| Agent loop | Plan, act, verify | 控制器运行 LLM 调用与工具调用交错，直到任务完成。 |
| Prompt injection | User attacks prompt | 试图覆盖系统提示词的恶意输入。 |

## Further Reading

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 晚期神经聊天机器人论文，就在 LLM 智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产指导，在 2026 年仍然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使提示注入成为首要安全问题的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用的编排层防御，包括 Plan-Verify-Execute 和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 来自间接提示注入的典型零点击数据泄露 CVE。说明为什么具有写访问权限的智能体需要运行时防御的参考案例。
