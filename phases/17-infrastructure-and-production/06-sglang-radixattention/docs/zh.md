# SGLang 与 RadixAttention：前缀密集型工作负载

> SGLang 将 KV cache 视为一等可复用资源，存储在 radix tree（基数树）中。vLLM 使用 FCFS（先到先服务）调度请求，而 SGLang 的 cache-aware scheduler（缓存感知调度器）优先处理具有更长共享前缀的请求——实际上是对最热的分支进行深度优先遍历，使其常驻 HBM。在 Llama 3.1 8B 上，使用类似 ShareGPT 的 1K prompt，SGLang 达到约 16,200 tok/s，而 vLLM 约为 12,500，领先约 29%。在前缀密集型 RAG 工作负载上，优势可达 6.4 倍。在语音克隆型工作负载上，缓存命中率超过 86%。2026 年已部署在超过 40 万个 GPU 上，覆盖 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS。陷阱在于，当前缀顺序不一致时，6.4 倍的数字会消失——顺序是工程师的杠杆。

**类型：** 学习
**语言：** Python（标准库，玩具级 radix tree 缓存 + cache-aware 调度器）
**前置知识：** 第 17 阶段 · 04（vLLM 推理服务内部原理）、第 14 阶段（Agentic RAG）
**时间：** ~75 分钟

## 学习目标

- 绘制 RadixAttention 示意图：前缀如何存储在 radix tree（基数树）中，以及 KV block 如何在同一分支根下的序列之间共享。
- 解释 cache-aware scheduling（缓存感知调度）以及为什么 FCFS 在前缀密集型流量上是错误的。
- 给定前缀缓存命中率和 prompt 长度分布，计算工作负载的预期加速比。
- 说出使 6.4 倍数字成真而非错失的 prompt-ordering discipline（提示排序规范）。

## 问题

经典推理将每个请求的 prompt 视为不透明。即使 5,000 个 RAG 请求都以相同的 2,000 token 系统提示加相同的检索前言开头，vLLM 也会对该 2,000 token 前缀进行 5,000 次 prefill（预填充）。GPU 反复做相同的工作。

观察：agentic 和 RAG 工作负载中的 prompt 几乎总是共享长前缀。系统提示、工具 schema、few-shot 示例、检索头、对话历史——都在请求之间重复。如果你存储一次该前缀的 KV cache 并复用它，就无需再次 prefill。

RadixAttention 正是这样做的。Token 被索引在 radix tree（基数树）中；每个节点拥有从其根路径上的 token 序列的 KV block。新请求遍历树：任何 token 匹配的节点复用该节点的 KV block。Prefill 成本与"新"后缀成正比，而非完整 prompt。

挑战在于调度。如果两个请求共享 2,000 token 前缀，第三个仅共享同一前缀的 200 token，你希望一起服务两个长共享请求，使长前缀常驻 HBM。FCFS 做相反的事——它先服务先到者，可能在下一个长前缀请求到达前驱逐热分支。

## 概念

### Radix tree 作为 KV 索引

Radix tree（紧凑 trie）存储 token 序列。每个节点拥有一个 token 范围及其 KV block。子节点将一个或多个 token 扩展序列。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

新请求进入，包含系统提示 + "Context: <doc A>" + "Question: Carol"。调度器遍历：系统前缀匹配（124 block 复用），doc-A 分支匹配（31 block 复用），然后仅分配 "Question: Carol" 的新 block（4 block）。Prefill 成本：4 block 的新 token。没有树：160 block。Prefill 节省约 40 倍。

### Cache-aware scheduling（缓存感知调度）

Radix tree 支持的复用如果缓存颠簸就毫无意义。两个关键策略：

1. **Depth-first dispatch（深度优先调度）**。从队列中选择下一个请求时，优先选择与当前运行集根于同一分支的请求。这使热分支保持常驻。
2. **Branch-level LRU，而非 block-level LRU**。驱逐整个分支（从最短的未使用叶子开始），而非单个 block，使缓存形状匹配 radix 形状。

FCFS 违反两者。一个共享 2,000 token 的请求坐在一个共享 50 token 的请求后面，然后 2,000 token 分支被驱逐以接纳 50 token 的请求。

### 你应该记住的基准数字

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500 (~29% 领先)。
- 前缀密集型 RAG（相同系统 + 相同文档，不同问题）：SGLang 上最高 6.4 倍。
- 语音克隆工作负载：86.4% 前缀缓存命中率。
- SGLang 客户生产命中率：取决于 prompt 规范，50-99%。
- 2026 年已部署在超过 400,000 个 GPU 上。

### 排序陷阱

6.4 倍数字依赖于一致的 prompt-template ordering（提示模板排序）。如果客户端在某些请求中将 prompt 构造为 `[system, tools, context, history, question]`，而在其他请求中为 `[system, context, tools, history, question]`，树无法找到共享前缀。对人类来说看似共享前缀的东西，对 radix tree 来说是两个不同的序列。

工程师的杠杆：你的 prompt template 就是缓存键。固定顺序。将所有不可变内容（system、tools、schema）放在前面。将检索上下文放在其次。将用户问题放在最后。不要将动态内容穿插到前缀中。

研究中的真实案例：将动态内容移出可缓存前缀，使一个部署的缓存命中率从 7% 提升到 74%，仅用一个改动。

### RadixAttention 何时赢、何时输

赢：
- RAG（相同检索前言，不同问题）。
- Agent（相同工具 schema，不同查询）。
- 带长系统提示的聊天。
- 带重复前言的语音/视觉工作负载。

输（回到 vLLM 级吞吐量）：
- 单次生成，prompt 唯一（代码补全、无系统提示的开放式聊天）。
- 每个请求将唯一内容穿插到前缀中的动态 prompt。

### 为什么这是调度器问题，而不仅是 kernel 问题

你可以将 KV 复用实现为 kernel 技巧。SGLang 的洞察在于，只有当调度器保持热分支常驻时，复用才有回报。朴素的"如果可用则复用"策略会在混合负载下导致缓存颠簸。Radix tree 索引的调度器是将 kernel 技巧转化为 29% 生产优势的关键。

### 与 vLLM 的交互

两个系统并非严格竞争。2026 年 vLLM 增加了前缀缓存（`--enable-prefix-caching`）和 cache-aware router（Rust 编写的 vLLM Router）。差距缩小但未完全消失——SGLang 的整个栈是 radix-first；vLLM 是嫁接上去的。对于前缀复用主导的工作负载，SGLang 仍是默认选择。对于没有强前缀模式的通用推理，vLLM 持平或更好。

## 使用它

`code/main.py` 实现了一个玩具级 radix tree KV cache 加两种策略的调度器：FCFS 和 cache-aware。在相同工作负载上运行两者，报告前缀缓存命中率和吞吐量增量。然后运行"打乱顺序"工作负载以展示 6.4 倍的崩塌。

## 交付它

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（prompt-template 形状、检索模式、并发租户数），产出 prompt-ordering 处方和 SGLang 采用的建议/不建议。

## 练习

1. 运行 `code/main.py`。在相同工作负载上比较 FCFS 和 cache-aware。增量来自哪里——prefill 节省、decode 节省还是队列延迟？
2. 修改工作负载使 prompt 随机排列 `[system, tools, context]`。重新运行。命中率发生什么？为什么？
3. 计算在 Llama 3.1 8B 上将 2,000 token 系统提示作为 radix 分支常驻的 HBM 成本。与无前缀复用的 16 序列 batch 成本比较。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么树形 LRU 驱逐在前缀密集型负载下击败 block 形 LRU。
5. 客户报告仅 8% 缓存命中率。说出三个可能原因和你会运行的诊断。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| RadixAttention | "SGLang 的东西" | KV cache 索引为 radix tree，使共享前缀复用 block |
| Radix tree | "紧凑 trie" | 每个节点拥有 token 范围及其 KV block 的树 |
| Cache-aware scheduler | "热分支优先" | 优先调度共享常驻分支的请求的调度器 |
| 前缀缓存命中率 | "多少 prompt 是免费的" | 从复用 KV block 服务的 prompt token 比例 |
| FCFS | "先到先服务" | 破坏前缀局部性的默认调度 |
| Branch-level LRU | "驱逐叶子" | 与 radix 形状匹配的驱逐策略 |
| Prompt template ordering | "缓存键" | Prompt 的组件顺序决定树能共享什么 |
| System prompt pinning | "常驻前缀" | 保持不可变系统部分常驻以避免驱逐颠簸 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) —— 源码和文档。
- [SGLang documentation](https://sgl-project.github.io/) —— RadixAttention 和调度细节。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) —— 设计参考。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) —— 基准数字和调度器原理。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) —— vLLM 自己的类 radix 实现，供对比。
