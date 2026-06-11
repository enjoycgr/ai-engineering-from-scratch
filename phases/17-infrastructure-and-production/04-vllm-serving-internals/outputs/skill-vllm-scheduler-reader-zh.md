---
name: vllm-scheduler-reader
description: 通过读取调度器级旋钮诊断 vLLM 服务配置，识别 PagedAttention、continuous batching 和 chunked prefill 中哪个是瓶颈。
version: 1.0.0
phase: 17
lesson: 04
tags: [vllm, paged-attention, continuous-batching, chunked-prefill, serving, scheduler]
---

给定 vLLM 服务配置（模型、dtype、硬件、`--gpu-memory-utilization`、`--max-num-batched-tokens`、`--enable-chunked-prefill`、`--speculative-model` 或 `--speculative-config`、最大并发，以及观测到的 TTFT 平均/P99、ITL 平均/P99、吞吐量 tok/s 指标集），产出调度器级诊断。

产出：

1. 配置读取。对每个 flag，说明它控制的调度器行为和 2026 年默认值。标记任何设为非默认值的 flag 并说明原因。
2. 瓶颈识别。将瓶颈分类为以下之一：PagedAttention 配置不足（KV block 饥饿）、continuous-batching 停滞（WAITING 队列增长）、chunked-prefill 尺寸不当（TTFT 尾部尖峰）、decode 计算受限（ITL 下限）、或 HBM 受限（无法容纳 batch）。用报告的指标说明理由。
3. 旋钮推荐。具体、有序的举措——翻转哪个 flag、尝试哪个值、关注哪个指标。不要在没有先耗尽调度器级调优的情况下建议"加更多 GPU"。
4. 兼容性检查。针对 vLLM v0.18.0：将 `--enable-chunked-prefill` + `--speculative-model` 组合标记为硬不兼容。如果两者都需要，推荐 V1 中的 N-gram GPU speculative decoding 作为文档记录的例外。
5. 下一步阅读。根据诊断结果指向 vLLM v0.18.0 release notes、PagedAttention 论文或 Aleksa Gordic V1 scheduler 讲解之一。

硬性拒绝：
- 没有四个核心指标（TTFT、ITL、吞吐量、并发）就进行诊断。拒绝并要求提供指标集。
- 未检查 speculative-decoding 配置就推荐 `--enable-chunked-prefill`。
- 将 `DCGM_FI_DEV_GPU_UTIL` 当作扩缩容信号。vLLM 会预分配 KV；占空比数字具有误导性。

拒绝规则：
- 如果 H100 上报告的吞吐量低于 100 tok/s，瓶颈很可能不是 vLLM——检查客户端 tokenizer、Python GIL 或请求级序列化。
- 如果 `--gpu-memory-utilization` 设为低于 0.7，拒绝进一步调优——操作者选择将 HBM 闲置，修复方案是在翻转调度器 flag 之前先提高上限。
- 如果操作者要求 draft-model speculation 下的 speculative-decoding + chunked-prefill 配方，拒绝并说明 v0.18.0 不兼容。指向第 17 阶段 · 05 的 EAGLE-3。

输出：一页调度器诊断，列出 flag、瓶颈、有序推荐、兼容性说明和下一步阅读指针。结尾用"下一步测量什么"段落，根据识别的瓶颈命名 P99 ITL、block 分配率或 WAITING 队列深度之一。
