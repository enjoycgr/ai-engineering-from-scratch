---
name: skill-rag-pipeline
description: 从第一性原理构建和调试 RAG 流水线
version: 1.0.0
phase: 11
lesson: 6
tags: [rag, retrieval, embeddings, vector-search, llm-engineering]
---

# RAG 流水线范式

每个 RAG 系统都遵循这一范式：

```
documents -> chunk -> embed -> store
query -> embed -> search(top_k) -> build_prompt -> generate
```

索引每个文档运行一次。查询在每个用户请求时运行。

## 何时使用 RAG

- LLM 需要访问私有或最近的文档
- 微调太贵或更新太慢
- 你需要为回答引用来源
- 知识库频繁变更

## 何时不使用 RAG

- 答案是 LLM 已有的通用知识
- 任务是创造性的（写作、头脑风暴）而非 factual 的
- 你需要模型采用特定的推理风格（使用微调）

## 实现检查清单

1. 将文档分块为 256-512 token 的段落，50 token 重叠
2. 使用一致的嵌入模型嵌入每个 chunk
3. 将嵌入存储在向量数据库中，保留原始文本
4. 查询时，使用相同模型嵌入用户问题
5. 通过余弦相似度检索 top-k（5-10）最相似的 chunk
6. 构建提示：系统指令 + 检索到的上下文 + 用户问题
7. 生成回答，将其 grounding 在检索到的上下文中
8. 返回带来源引用的回答

## 常见错误

- 索引和查询使用不同的嵌入模型（向量不兼容）
- Chunk 太小（丢失上下文）或太大（稀释相关性）
- chunk 之间没有重叠（在边界处分割句子）
- 文档变更时忘记重新索引
- 不向用户生成连贯回答，直接返回检索到的 chunk
- 对 factual RAG 查询没有设置 temperature=0（更高的 temperature = 更多幻觉）

## 调试检索

如果正确的 chunk 没有被检索到：
1. 打印查询嵌入并验证其非零
2. 手动检查已知相关 chunk 的余弦相似度
3. 尝试改写查询以匹配文档词汇
4. 验证索引和查询时的嵌入模型匹配
5. 检查相关内容是否在分块过程中丢失

## 生产参数

- Chunk 大小：256-512 tokens
- 重叠：50 tokens（chunk 大小的 10-20%）
- Top-k：大多数用例 5-10
- Temperature：factual 回答设为 0
- 嵌入模型：text-embedding-3-small（性价比高）或 text-embedding-3-large（更高准确率）
