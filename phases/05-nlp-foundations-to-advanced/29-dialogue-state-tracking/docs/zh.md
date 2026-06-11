# 对话状态追踪 (Dialogue State Tracking)

> "我想找一家北边便宜的餐厅……其实改成中等价位……再加意大利菜。" 三轮对话，三次状态更新。DST 保持 slot-value 字典同步，以便完成预订。

**类型:** Build
**语言:** Python
**前置知识:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**时间:** ~75 分钟

## 问题定义

在任务导向型对话系统中，用户的目标被编码为一组 slot-value 对：`{cuisine: italian, area: north, price: moderate}`。每一轮用户对话都可能增加、修改或删除某个 slot。系统必须阅读整段对话并正确输出当前状态。

只要有一个 slot 出错，系统就会订错餐厅、排错航班或刷错卡。DST 是用户所说内容与后端执行之间的关键枢纽。

为什么到 2026 年 LLM 时代 DST 仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的 slot 值，而非自由文本生成。
- 工具使用型智能体（tool-use agents）在调用 API 前仍需要 slot 解析。
- 多轮修正比看起来更难："其实不对，改成周四吧。"

现代流水线：经典 DST 概念 + LLM 抽取器 + 结构化输出护栏。

## 核心概念

![DST: 对话历史 → slot-value 状态](../assets/dst.svg)

**任务结构。** schema 定义 domain（restaurant、hotel、taxi）及其 slot（cuisine、area、price、people）。每个 slot 可以为空、填入来自封闭集合的值（price: {cheap, moderate, expensive}），或自由文本值（name: "The Copper Kettle"）。

**两种 DST 形式。**

- **分类（Classification）。** 对每个 (slot, candidate_value) 对预测 yes/no。适用于封闭词表的 slot。2020 年前的标准做法。
- **生成（Generation）。** 给定对话，以自由文本生成 slot 值。适用于开放词表的 slot。现代默认做法。

**评估指标。** Joint Goal Accuracy (JGA, 联合目标准确率) —— 所有 slot 都正确的轮次占比。全对或全错。MultiWOZ 2.4 排行榜在 2026 年最高约 83%。

**架构。**

1. **基于规则（slot 正则 + 关键词）。** 窄域强基线。可调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制生成。LLM 出现前的标准方案。
3. **LDST (LLaMA + LoRA)。** 指令微调的 LLM，带 domain-slot 提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别质量。
4. **无本体（Ontology-free, 2024–26）。** 跳过 schema；直接生成 slot 名称和值。可处理开放域。
5. **提示 + 结构化输出（Prompt + structured output, 2024–26）。** LLM + Pydantic schema + 约束解码。5 行代码，生产就绪。

### 经典失效模式

- **跨轮共指（Co-reference across turns）。** "我们就选第一个选项吧。" 需要解析指的是哪个选项。
- **覆盖 vs 追加。** 用户说 "加意大利菜。" 是替换 cuisine 还是追加？
- **隐式确认。** "好的没问题" —— 这是否接受了推荐的预订？
- **修正（Correction）。** "其实改成 7 点。" 必须只更新 time，不清除其他 slot。
- **指向前一轮系统回复的共指。** "是的，就那个。" 哪个"那个"？

## 动手实现

### 步骤 1：基于规则的 slot 抽取器

见 `code/main.py`。正则 + 同义词字典可覆盖窄域中 70% 的规范表达：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词表之外较脆弱。适用于确定性 slot 确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三条不变式：

- 绝不重置用户未提及的 slot。
- 显式否定（"不用管 cuisine 了"）必须清除。
- 用户修正（"其实……"）必须覆盖，而非追加。

### 步骤 3：基于 LLM 的结构化输出 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证输出有效的状态对象。无需正则、无 schema 不匹配、无幻觉 slot。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准标准：系统有多少比例的轮次把**所有** slot 都答对了？对于 MultiWOZ 2.4，2026 年顶级系统为 80-83%。你的窄域系统应在限定词表上超过该值，否则 LLM 基线会击败你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最近更新的 slot，而非追加。没有 LLM 辅助很难完全做对。现代模式：始终让 LLM 从历史中重新生成完整状态，而不是增量更新 —— 这天然就能处理修正。

## 常见陷阱

- **全历史再生成本。** 每轮都让 LLM 重新生成状态的代价是 O(n²) 总 token。应限制历史长度或摘要旧轮次。
- **Schema 漂移（Schema drift）。** 事后添加新 slot 会破坏旧训练数据。为你的 schema 做版本管理。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN" —— 处处做规范化。
- **隐式继承。** 如果用户之前指定了 "4 个人"，新的不同时间请求不应清除 people。始终传入完整历史。
- **自由文本 vs 封闭集合。** 名称、时间和地址需要自由文本 slot；菜系和地区是封闭的。在 schema 中混用两者。

## 应用场景

2026 年技术栈：

| 场景 | 方案 |
|-----------|----------|
| 窄域（一两个意图） | 基于规则 + 正则 |
| 宽域，有标注数据 | LDST (LLaMA + LoRA 在 MultiWOZ 风格数据上微调) |
| 宽域，无标注，生产就绪 | LLM + Instructor + Pydantic schema |
| 语音 / 口语 | ASR + 归一化 + LLM-DST |
| 多域预订流程 | 基于 schema 的 LLM，每域一个 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 兜底并带确认流程 |

## 交付物

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: 设计对话状态追踪器 —— schema、提取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定一个用例（domain（领域）、语言、词汇开放度、合规需求），输出：

1. Schema。domain 列表、每个 domain 的 slot、每个 slot 的开放 vs 封闭词汇。
2. Extractor（提取器）。基于规则 / seq2seq / LLM-with-Pydantic。给出理由。
3. Update policy（更新策略）。全状态重新生成 / 增量更新；修正处理；否定处理。
4. Evaluation（评估）。在留出对话集上的 Joint Goal Accuracy、slot 级别 precision/recall、最难 slot 的混淆情况。
5. Confirmation flow（确认流程）。何时明确请求用户确认（破坏性操作、低置信度提取）。

拒绝无基于规则的二次检查的、用于合规敏感 slot 的纯 LLM DST。拒绝无法回滚用户修正的 slot 的任何 DST。标记无版本号的 schema。
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个 slot（cuisine、area、price）构建基于规则的状态追踪器。在 10 段手工编写的对话上测试。测量 JGA。
2. **中等。** 在同一数据集上使用 Instructor + Pydantic + 小型 LLM。对比 JGA。检查最难的轮次。
3. **困难。** 实现两种方案并路由：基于规则为主，当规则输出 <2 个高置信度 slot 时 fallback 到 LLM。测量联合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | 在对话轮次之间维护 slot-value 字典。 |
| Slot | 用户意图的单位 | 后端需要的具名参数（cuisine、date）。 |
| Domain | 任务领域 | Restaurant、hotel、taxi —— slot 的集合。 |
| JGA | Joint Goal Accuracy | 所有 slot 都正确的轮次占比。全对或全错。 |
| MultiWOZ | 基准数据集 | 多领域 WOZ 数据集；标准 DST 评估基准。 |
| Ontology-free DST | 无 schema | 直接生成 slot 名称和值，无固定列表。 |
| Correction | "其实……" | 覆盖先前已填充 slot 的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调用于 DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力方案。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 经典 DST 结果。
