# vLLM 推理服务内部原理：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位建立在三个叠加的默认特性上，而非单一技巧。PagedAttention 始终开启。Continuous batching 在每次 decode 迭代之间将新请求注入活跃 batch。Chunked prefill 将长提示切片，使 decode token 永远不会被饿死。三者同时开启时，Llama 3.3 70B FP8 在单张 H100 SXM5 上、128 并发下可达到 2,200–2,400 tok/s，比 vLLM 自身默认配置高约 25%，是朴素 PyTorch 循环的 3–4 倍。本课以可绘图的方式讲解调度器和 attention kernel，最后以 `code/main.py` 中的玩具级 continuous batcher 收尾，它按 vLLM 的方式调度 prefill 和 decode。

**类型：** 学习
**语言：** Python（标准库，玩具级 continuous batching 调度器）
**前置知识：** 第 17 阶段 · 01（模型服务）、第 11 阶段（LLM 工程）
**时间：** ~75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV cache 分配器：block、block table，以及为什么在生产负载下碎片率保持在 4% 以下。
- 在迭代级别绘制 continuous batching 的示意图：已完成的序列如何离开 batch，新序列如何无需排空即可加入。
- 用一句话描述 chunked prefill，并说出它保护的延迟指标（提示：是 TTFT 尾部，而非平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 中同时开启所有优化时会踩到的坑。

## 问题

朴素的 PyTorch 服务循环一次只处理一个请求：tokenize、prefill、decode 到 EOS、返回。一个用户时没问题。一百个用户时，就变成了一群有耐心的人在排队。显而易见的修复——静态 batching——将每个请求填充到窗口中最长的 prompt，将每次 decode 填充到最长预期输出，并在最慢的序列上卡住整个 batch。你为你从未使用的填充付费，快的请求等慢的请求。

vLLM 同时解决了三个问题。PagedAttention 阻止了 KV cache 碎片化，经典连续分配会吃掉 60–80% 的 GPU 内存。Continuous batching 让请求在每次 decode 迭代之间加入和离开 batch，因此 batch 始终充满实际工作。Chunked prefill 将 32k token 的 prompt 切成约 512 token 的切片，与 decode 交错执行，因此长 prompt 不会冻结 GPU 上所有其他 decode token。

2026 年的生产默认是三者全开。你需要理解每个特性的作用，因为故障模式都在调度器上，而非模型上。

## 概念

### PagedAttention 作为虚拟内存系统

KV cache 的体积为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` / 序列。对于 Llama 3.3 70B 在 8192 token 下，BF16 约为 1.25 GB/序列。如果你为每个请求预预留 8192 个槽位，但平均请求只用了 1500 token，你浪费了约 82% 的预留 HBM。经典 batching 为此买单。

PagedAttention 借鉴了操作系统虚拟内存的思想。KV cache 不是按序列连续存放的。它以固定大小的 block（默认 16 token）分配。每个序列有一个 block table，将其逻辑 token 位置映射到物理 block ID。当序列增长超出已分配 block 时，增加一个 block。当序列结束时，其 block 归还到池中。

碎片率从 60–80%（经典）降到 4% 以下（PagedAttention）。你不需要用 flag 开启 PagedAttention——它是 vLLM 唯一的分配器。可调参数是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后，为 KV block 预留多少 HBM。

### Continuous batching 在迭代级别

旧的"dynamic batching"等待一个窗口（比如 10 ms）填满 batch，然后运行 prefill + decode + decode + decode 直到每个序列完成。快的序列提前离开，在 GPU 完成慢的序列时空闲。

Continuous batching 在每次 decode 步骤之间操作。将正在运行的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚命中 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV block，它接纳新序列（prefill 或恢复）。
3. 前向传播在 `RUNNING` 中的内容上运行，每个序列发出一个新 token。

Batch size 从不填充到固定数量。处于不同输出位置的序列共享一次融合前向。在 2026 年 vLLM 中这被称为 `V1 scheduler`。关键不变量：调度器每次 decode 迭代运行一次，而非每次请求运行一次。

### Chunked prefill 保护 TTFT 尾部

Prefill 是计算密集型的。Llama 3.3 70B 上 32k token 的 prompt 在单张 H100 上纯 prefill 约需 800 ms。Prefill 运行时，batch 中所有其他序列的 decode token 都在等待。在服务循环中，一个长 prompt 的首 token 延迟（TTFT）会变成几十个其他用户的 token 间延迟（ITL）尖峰。

Chunked prefill 将 prefill 拆成固定大小的 chunk（默认 512 token），每个 chunk 作为一个调度单元。Chunk 之间，调度器可以让 decode 序列前进一个 token。你以微小的绝对 prefill 延迟代价（每个 chunk 几 ms），换取低得多的 decode 时间抖动。在公开基准中，混合负载下的 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个默认特性的交互

三个特性相互依赖。PagedAttention 给调度器提供了细粒度的 KV 资源来权衡。Continuous batching 需要这种细粒度资源，因此接纳新序列不会强制全局重排。Chunked prefill 是调度器在同一个 `RUNNING` 列表上做出的决策——它是调度器策略的一部分，而非独立系统。

你不需要知道每个 flag。你需要知道调度器优化什么：在 KV block 预算约束下的 goodput，受 chunked prefill 切片影响。

### 2026 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与 draft-model speculative decoding（`--speculative-model`）组合使用。文档中记录的例外是 V1 scheduler 中的 N-gram GPU speculative decoding。不读 release notes 就把所有 flag 打开的团队会在启动时遇到运行时错误，而非软回归。如果你的投机收益值得为此启用 chunked prefill，请重新审视选择——2026 年的正确答案通常是 EAGLE-3 而不开 chunked prefill，而非 draft model 加 chunked prefill 导致无法编译。

### 你应该记住的数字

- Llama 3.3 70B FP8, H100 SXM5, 128 并发, 三者全开: 2,200–2,400 tok/s。
- 同模型, vLLM 默认（无 chunked prefill）: ~1,800 tok/s。
- 同模型, 朴素 PyTorch 前向循环: ~600 tok/s。
- PagedAttention 在生产负载下的 KV 碎片浪费: <4%。
- 混合负载下的 P99 ITL: ~15 ms（有 chunked prefill）, ~50 ms（无）。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个循环的纯 Python 实现，使用假 token 计数和假前向延迟。运行它可以看到 chunked prefill 如何在长 prefill 期间保持 decode 序列活跃。

## 使用它

`code/main.py` 模拟了一个带可切换特性的 vLLM 风格调度器。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无 batching。
- `STATIC` 模式：填充并等待，经典 batching。
- `CONTINUOUS` 模式：迭代级准入和释放。
- `CONTINUOUS + CHUNKED` 模式：prefill 切片与 decode 交错。

输出显示总吞吐量（每秒虚拟 token）、平均 TTFT 和 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 行应该占主导。

## 交付它

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定一个服务配置（batch size、KV 内存利用率、chunked prefill 大小、speculative 配置），它产出调度器诊断，指出三个默认特性中哪个是瓶颈以及该调什么。

## 练习

1. 运行 `code/main.py`。在短请求和长请求混合的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里——prefill 效率、decode 效率还是尾部延迟？
2. 修改玩具调度器，添加 `--max-num-batched-tokens`。H100 上运行 Llama 3.3 70B FP8 时，合适的值是多少？（提示：它是 KV block 大小和空闲 block 数量的函数，而非原始 HBM。）
3. 重读 vLLM v0.18.0 release notes。哪些 flag 组合互斥？列出它们。
4. 计算 1000 个请求的 KV cache 碎片浪费，平均输出 1500 token，标准差 600 token，在（a）8192 最大值的连续单请求分配，（b）16-token block 的 PagedAttention 下。
5. 用一段话解释为什么 chunked prefill 帮助 P99 ITL 但不单独提高平均吞吐量。实践中吞吐量提升来自哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| PagedAttention | "KV 技巧" | KV cache 的固定大小 block 分配器；碎片率 <4% |
| Block table | "页表" | 每序列从逻辑 token 位置到物理 KV block 的映射 |
| Continuous batching | "dynamic batching，但正确" | 每次 decode 迭代做出准入/释放决策 |
| Chunked prefill | "prefill 拆分" | 将长 prefill 拆成 512-token 切片，与 decode 交错 |
| TTFT | "首 token 时间" | Prefill + 排队 + 网络；长 prompt 下由 prefill 主导 |
| ITL | "token 间延迟" | 连续 decode token 之间的时间；由 batch size 主导 |
| Goodput | "满足 SLO 的吞吐量" | 每个请求仍命中 TTFT 和 ITL 目标的 tok/s |
| V1 scheduler | "新调度器" | vLLM 2026 年调度器；N-gram spec decode 是与 chunked prefill 兼容的路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 加载权重和激活后，为 KV block 预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) —— chunked-prefill 和 speculative-decoding 兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) —— 2026 年发布节奏和版本特定行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) —— 定义如何思考分配器的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) —— 碎片分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) —— 带火焰图的详细 V1 scheduler 讲解。
