---
name: fipa-mapper
description: 将任意2026年的agent-protocol (智能体协议)规范（MCP、A2A、ACP、ANP、CA-MCP、NLIP或新规范）映射到FIPA-ACL performatives (施事语)和interaction protocols (交互协议)，以判断哪些是真正的创新，哪些是重新发明。
version: 1.0.0
phase: 16
lesson: 02
tags: [multi-agent, protocols, FIPA, speech-acts, interoperability]
---

给定一个新的agent-protocol (智能体协议)规范，生成FIPA-ACL映射，使读者能够分辨哪些部分是重新发明，哪些是真正的全新结构。

生成内容：

1. **Envelope mapping (信封映射)。** 对于规范定义的每种消息类型，命名最接近的FIPA performative (施事语)（`inform`、`request`、`query-if`、`query-ref`、`propose`、`accept-proposal`、`reject-proposal`、`cfp`、`subscribe`、`cancel`、`failure`、`not-understood`，或其他约20种）。如果没有合适的performative (施事语)，精确描述差距。
2. **Correlation model (关联模型)。** 该规范如何将请求与回复关联、将取消与原始请求关联、将流式事件与订阅关联？与FIPA的`:conversation-id`和`:reply-with`字段进行比较。
3. **Content-language stance (内容语言立场)。** 该规范是否强制要求content schema (内容模式)（typed artifacts (类型化工件)、JSON-Schema）、接受natural language (自然语言)，还是保持开放？与FIPA的SL0/SL1和ontology (本体论)字段进行比较。
4. **Interaction-protocol library (交互协议库)。** 哪些FIPA interaction protocols (交互协议)可以在该规范之上实现：contract-net (合同网)、subscribe-notify (订阅-通知)、request-when (条件请求)、propose-accept (提议-接受)？命名实现每种协议所需的消息。
5. **Discovery model (发现模型)。** agent (智能体)如何找到对应方和能力（MCP的`listTools`、A2A的Agent Card、ANP的DID + meta-protocol）？与FIPA的directory facilitator (目录协调器)和yellow-pages service (黄页服务)进行比较。
6. **Reinvention vs novelty (重新发明 vs 创新)。** 生成一个简短的表格，三列：[FIPA概念、现代规范等价物、变化内容]。每行标记为[reinvention (重新发明)]或[novel-structure (新结构)]。只有当规范引入了FIPA没有的primitive (原语)时，该行才是"novel-structure (新结构)"——decentralized identity (去中心化身份)、typed multimodal artifacts (类型化多模态工件)和LLM-interpretable content (LLM可解释内容)是常见的候选。

Hard rejects (硬性拒绝)：

- 任何声称某个规范是"revolutionary (革命性)"但未展示FIPA没有的primitive (原语)的映射。Speech-act theory (言语行为理论) + ontology overhead (本体论开销)是失败模式，而非primitives (原语)本身。
- 忽略discovery layer (发现层)的framework comparisons (框架比较)。没有discovery (发现)的规范是不完整的，而非创新的。
- 诸如"Protocol X replaces FIPA (协议X取代FIPA)"的表述，未解决两个agent对内容含义产生分歧时会发生什么（semantic drift (语义漂移)）。

Refusal rules (拒绝规则)：

- 如果规范处于pre-standardization (预标准化)阶段（草案 < 6个月，无公开实现），说明映射是provisional (临时性的)，并标出三个最可能的变化。
- 如果规范是closed-source (闭源)或enterprise-only (仅限企业)（某些ACP变体），映射已记录的内容并指出差距。
- 如果用户只提供了一篇blog post (博客文章)（无规范文档），要求在映射前先提供规范。

Output (输出)：一页brief (简报)。以single-sentence summary (单句摘要)开头（"Protocol X is FIPA `request`/`subscribe` with JSON syntax and a DID-based discovery layer."），然后是上述六个部分，最后是一段closing paragraph (收尾段落)，回答："Which old FIPA failure mode will this spec rediscover?"
