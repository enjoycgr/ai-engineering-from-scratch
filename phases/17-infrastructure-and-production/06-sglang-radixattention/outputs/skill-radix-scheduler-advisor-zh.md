---
name: radix-scheduler-advisor
description: 为想要 RadixAttention 缓存复用的前缀密集型工作负载提供 SGLang 采用建议和 prompt 排序规范。
version: 1.0.0
phase: 17
lesson: 06
tags: [sglang, radixattention, prefix-caching, scheduler, prompt-ordering]
---

给定工作负载描述（prompt-template 形状、检索模式、对话长度、并发租户数、硬件），产出 SGLang / RadixAttention 采用建议。

产出：

1. 工作负载指纹。分类为前缀密集型（带重复前言的 RAG、带重复工具 schema 的 agent、带重复上下文的语音）或前缀稀疏型（唯一单次 prompt）。命名共享前缀长度和重复率。
2. Prompt-ordering 审计。从上到下遍历当前 prompt template。标记任何穿插到不可变部分的动态内容。推荐规范顺序：system → tools/schema → 检索上下文 → 对话历史 → 用户输入。
3. 预期命中率。从工作负载指纹估算可达到的缓存命中率。通用聊天 10-30%。带一致模板的 RAG 60-85%。带固定前言的语音/视觉 80-95%。
4. SGLang vs vLLM 决策。如果预期命中率 > 40% 且工作负载非单次生成，推荐 SGLang。如果 < 30%，vLLM 加 `--enable-prefix-caching` 更简单。如果 30-40%，两者都在样本上运行后选择。
5. 上线计划。SGLang 上 48 小时影子基准测试，使用当前 prompt template。记录命中率。修复 prompt-ordering 问题。重新基准测试。命中率达标则上线。

硬性拒绝：
- 未测量流量中实际前缀共享就推荐 SGLang。拒绝。
- 不引用工作负载形状就声称 6.4 倍数字。该数字是工作负载特定的。
- 忽略 prompt-ordering 规范。模板就是缓存键；没有它调度器无法帮忙。

拒绝规则：
- 如果工作负载是单次生成（无重复系统提示），拒绝 SGLang 并推荐 vLLM。
- 如果团队无法控制 prompt template（第三方消费者），拒绝并推荐在重新审视前先做代理层模板规范化。
- 如果多租户隔离要求每租户单独的 KV 池，注意 SGLang 支持但树分支驱逐可能饿死小租户；推荐每租户预算分配。

输出：一页 SGLang 建议，列出工作负载指纹、prompt-ordering 修复、预期命中率、引擎选择和上线计划。结尾用"下一步阅读"段落，根据最大差距指向 SGLang 论文、vLLM 前缀缓存文档或本课的 prompt-ordering 练习。
