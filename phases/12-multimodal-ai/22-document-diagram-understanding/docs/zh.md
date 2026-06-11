# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表格有布局、表格、图表、脚注、页眉和语义结构，纯图像理解无法捕捉。VLM 之前的栈是流水线：Tesseract OCR + LayoutLMv3 + 表格提取启发式。VLM 浪潮用 OCR-free 模型取代——Donut (2022)、Nougat (2023)、DocLLM (2023)——直接发出结构化标记。到 2026 年前沿只是"在 2576px 原生下将页面图像喂给 Claude Opus 4.7"，结构化标记输出免费获得。本课阅读文档 AI 的三时代弧。

**类型：** Build
**语言：** Python (stdlib, layout-aware document parser skeleton)
**前置知识：** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**时间：** ~180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三个输入流：文本、布局（bbox）、图像 patch，带统一 masking。
- 对比 Donut（OCR-free，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成）、PaliGemma 2（VLM-native）。
- 为新任务（发票、科学论文、手写表格、中文收据）挑选文档模型。

## 问题

"理解这个 PDF"出奇地难。信息存在于：

- 文本内容（90% 信号）。
- 布局（页眉、脚注、边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图表和图示。
- 手写注释。
- 字体和排版（标题 vs 正文）。

原始 OCR 倾倒文本并丢失其余。在乎发票的系统需要知道"Total: $1,245"来自右下角，不是来自脚注。

## 概念

### 时代 1 — OCR 流水线（2021 年前）

经典栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取带每词边界框的文本。
3. 布局分析器识别块（页眉、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + regex 提取字段。

对干净印刷文本有效。在手写、倾斜扫描、复杂表格、非英语脚本上崩溃。每个失败模式需要自定义异常路径。

### TrOCR (2021)

TrOCR (Li 等人, arXiv:2109.10282) 用 transformer 编码器-解码器替代 Tesseract 的经典 CNN-CTC，在合成 + 真实文本图像上训练。手写和多语言文本的干净胜利。仍是流水线（检测器然后 TrOCR 然后布局），但 OCR 步骤大幅提升。

### 时代 2 — OCR-free（2022-2023）

第一批 OCR-free 模型说：完全跳过检测，将图像像素直接映射到结构化输出。

Donut (Kim 等人, arXiv:2111.15664)：
- 编码器-解码器 transformer，编码器是 Swin-B。
- 输出是表单理解的 JSON、摘要的 markdown 或任何任务特定 schema。
- 无 OCR、无布局、无检测。

Nougat (Blecher 等人, arXiv:2308.13418)：
- 专门为科学论文训练。
- 输出是 LaTeX / markdown。
- 处理方程、多栏布局、图表。
- 每个 arXiv 解析器调用的模型。

这些是专家，不是通才。Donut 在科学论文上失败；Nougat 在发票上失败。

### LayoutLMv3 (2022)

不同轨道。LayoutLMv3 (Huang 等人, arXiv:2204.08387) 保留 OCR 但添加布局理解：

- 三个输入流：OCR 文本 token、每 token 2D 边界框、图像 patch。
- 跨所有三种模态的 masked 训练目标（mask 文本、mask patch、mask 布局）。
- 下游：分类、实体提取、表格 QA。

LayoutLMv3 是 OCR 基础文档理解的巅峰。表单和发票上强。需要上游 OCR。VLM 前标准化文档基准上的最佳准确率。

### DocLLM (2023)

DocLLM (Wang 等人, arXiv:2401.00908) 是 LayoutLM 的生成兄弟。条件于布局 token 生成自由形式答案。文档 QA 更好；仍依赖 OCR 输入。

### 时代 3 — VLM-native（2024+）

2024 VLMs 变得足够好以完全取代流水线。以高分辨率将完整页面图像喂给 VLM，问问题，得到答案。

- LLaVA-NeXT 336-tile AnyRes 对小文档有效。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2 (2025 年 4 月) 专门为文档 + 手写训练。

VLM-native 与 OCR-流水线之间的差距快速关闭。到 2026 年，VLM-native 在以下方面获胜：

- 场景文本（手写 + 印刷、混合脚本）。
- 合并单元格的复杂表格。
- 嵌入文本中的数学方程。
- 带文本注释的图表。

OCR 流水线仍在以下方面获胜：

- 大规模纯扫描工作负载，每页延迟重要。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

2576 像素原生输入下，前沿 VLM 以接近人类准确率做文档理解。2026 年初基准数字：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，流水线 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

封闭模型差距主要在分辨率和基础 LLM 规模。7B 开放模型落后几分但在追赶。

### 数学方程和 LaTeX 输出

科学论文需要方程的精确 LaTeX 输出。Nougat 为此训练。带 LaTeX 目标的 VLMs（Qwen2.5-VL-Math、Nougat 衍生）产生可用 LaTeX。没有显式 LaTeX 训练，VLMs 产生可读但不精确的转录。

2026 年科学论文流水线：PDF 上链 Nougat，然后 VLM 处理棘手页面。

### 手写

仍是最难的子任务。混合印刷 + 手写（医生笔记、填写表格）是 OCR 流水线在成本上仍击败 VLMs 的地方。仅手写 VLMs 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 年配方

对新文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学 + 手写 + 表单）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄入：图表用 Nougat，VLM 处理图表。
- 监管：OCR 流水线 + VLM 验证器做交叉检查。

## 使用它

`code/main.py`：

- 玩具布局感知 tokenizer：给定（文本、bbox）对，产生 LayoutLMv3 风格输入。
- Donut 风格任务 schema 生成器：表单的 JSON 模板。
- 跨 OCR-流水线、Donut、Nougat 和 VLM-native 的每页 token 预算对比。

## 交付它

本课产生 `outputs/skill-document-ai-stack-picker.md`。给定文档 AI 项目（领域、规模、质量、监管），在 OCR 流水线、OCR-free 专家和 VLM-native 之间挑选。

## 练习

1. 你的项目是每天 1000 万发票。哪个栈最小化每页成本而不丢失准确率？

2. 为什么 LayoutLMv3 在表单 QA 上优于纯 CLIP-VLM 但在场景文本上落后？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出 VLM-native 输出在 LaTeX 保真度上击败 Nougat 的测试案例，以及 Nougat 获胜的案例。

4. 阅读 PaliGemma 2 论文 (Google, 2024)。什么关键训练数据添加提升了文档准确率 vs PaliGemma 1？

5. 设计监管安全混合：OCR 流水线作为主要，VLM 作为次要交叉检查。你如何解析分歧？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OCR pipeline | "Tesseract-style" | 阶段式栈：检测 -> OCR -> 布局 -> 规则；确定性、脆弱 |
| OCR-free | "Donut-style" | 跳过显式 OCR 的图像到输出 transformer；单一模型 |
| Layout-aware | "LayoutLM" | 输入包括每 token bbox 坐标；跨模态统一 masking |
| VLM-native | "Frontier VLM" | 以高分辨率直接将页面图像喂给 Claude/GPT/Qwen VLM；无流水线 |
| DocVQA | "Doc benchmark" | 文档 VQA 标准；最常引用的分数 |
| Markup output | "LaTeX / MD" | 结构化输出格式替代自由形式文本；启用下游自动化 |

## 延伸阅读

- [Li 等人 — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher 等人 — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang 等人 — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim 等人 — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang 等人 — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
