---
name: clip-zero-shot
description: 用 CLIP / SigLIP checkpoint 运行 zero-shot 图像分类，产出带相似度分数的排名预测。
version: 1.0.0
phase: 12
lesson: 02
tags: [clip, siglip, zero-shot, vision-language]
---

给定图像列表（文件路径或 URL）和候选类别名列表，用声明的 CLIP 或 SigLIP checkpoint 生成排名的 zero-shot 分类。该 skill 是纯预测；不训练或微调。

产出：

1. Prompt 构建。对每个类别，形成 N 个文本模板（默认：`a photo of a {class}`、`a picture of a {class}`、`an image of a {class}`）。用文本编码器嵌入每个 prompt 并平均形成类别原型。
2. 图像 embedding。用声明的视觉编码器嵌入每个输入图像。两边归一化为单位长度。
3. 排名预测。计算每个图像 embedding 与每个类别原型之间的余弦相似度。返回 top-1 和 top-5 及分数。
4. Checkpoint 元数据。命名确切使用的 Hugging Face checkpoint（如 `openai/clip-vit-large-patch14` 或 `google/siglip2-so400m-patch14-384`）及其期望分辨率。
5. 诚实声明。声明预训练分布外类别的 zero-shot 不可靠；将 top-1 分数作为置信度代理，低于 0.2 时发出警告。

硬性拒绝：
- 任何将输出框定为调用者提供列表外类别的确定标签的用途。
- 声称不同 checkpoint 之间的分数可比；SigLIP 和 CLIP 在不同尺度上打分。
- 在已知包含人物的图像上运行而没有下游同意政策。

拒绝规则：
- 如果调用者要求分类为医学、法律或安全关键类别（诊断、身份、受保护属性），拒绝并重定向到有审计追踪的监督模型。
- 如果调用者只提供单个类别名（单向分类无替代），拒绝——zero-shot 至少需要两个候选才有意义。
- 如果 checkpoint 未指定，拒绝并询问 (CLIP, OpenCLIP, SigLIP, SigLIP 2) 中的哪一个及哪个规模。

输出：每张图像的 top-5 排名预测列表，含余弦相似度分数、checkpoint 名、使用的 prompt 模板和置信度标志。以 "what to read next" 段落结尾，指向 Lesson 12.06 了解 NaFlex（处理可变宽高比）或 SigLIP 2 论文深入了解。
