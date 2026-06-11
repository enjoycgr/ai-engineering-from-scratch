---
name: llava-vibes-eval
description: 在 LLaVA 家族 VLM 上运行 10-prompt vibes-eval 并产出人类可读的记分卡。
version: 1.0.0
phase: 12
lesson: 05
tags: [llava, vlm, vibes-eval, instruction-tuning]
---

给定一个 LLaVA 家族 VLM（LLaVA-1.5、LLaVA-NeXT、LLaVA-OneVision 或社区分支）和测试图像集，运行覆盖标题生成、VQA、推理、拒绝和格式合规的 10-prompt 烟雾测试。产出确认 projector 和 LLM 正确连接的记分卡。

产出：

1. 十个带预期行为描述的 prompt：
   - 三个标题生成（短、详细、创意）。
   - 三个 VQA（计数、颜色、物体存在）。
   - 两个推理（比较两个区域、因果）。
   - 两个拒绝（私人个体、PII 识别）。
2. 每 prompt 分数。通过 / 部分 / 失败，附一行理由。
3. 整体模式诊断。如果标题通过但 VQA 失败，怀疑阶段 2 数据混合物。如果详细标题出现幻觉，怀疑 ShareGPT4V 风格数据不足。如果拒绝失败，标记安全数据缺口。
4. 分辨率检查。在 336x336 基础和 AnyRes 各运行一个需要 OCR 的 prompt；记录差异。低分辨率失败是预期的；高分辨率失败意味着 AnyRes 配置错误。
5. 建议跟进。如果特定类别失败，调用者可以运行的三个具体训练数据增补。

硬性拒绝：
- 没有同时运行 vibes 套件就按基准数字给 VLM 打分。基准可以被操纵；vibes 揭示真实部署准备度。
- 将幻觉与风格化冗长混为一谈。具体标记哪些物体是被发明的 vs 仅仅被详细描述。
- 声称推理 prompt 通过而不检查推理链，不只是最终答案。

拒绝规则：
- 如果调用者要求 vibes-eval 专有 VLM（Gemini、Claude、GPT-5V）而没有 API 访问，拒绝——测试需要实际推理。
- 如果目标用例是医学诊断或法律建议，拒绝——vibes-eval 不是认证，不能用于高风险领域。
- 如果没有提供图像，拒绝——测试按定义是图像 grounded 的。

输出：10 行记分卡（prompt、图像、预期、实际、通过/部分/失败）、整体模式诊断和三项目跟进列表。以 "what to read next" 段落结尾，指向 Lesson 12.06 (AnyRes) 了解分辨率相关失败或 Lesson 12.07 (ablations) 了解数据混合物调优。
