# AlphaEvolve — 进化式编程智能体

> 将一个前沿编程模型与一个进化循环和一个机器可检查的评估器配对。让循环运行足够长的时间。它发现了一个使用 48 次标量乘法的 4x4 复数矩阵乘法过程——这是 56 年来首次超越 Strassen 的成果。它还找到了一个 Google 全局 Borg 调度启发式算法，在生产环境中回收了约 0.7% 的集群计算资源。架构故意设计得平淡无奇。胜利来自评估器的严谨性。

**类型：** 学习
**语言：** Python（标准库，进化循环玩具）
**前置要求：** Phase 15 · 01（长程框架），Phase 15 · 02（自我教学推理）
**时间：** ~60 分钟

## 问题

大语言模型可以写代码。进化算法可以在代码上搜索。两者都已单独尝试了几十年；两者都遇到了天花板。LLM 的天花板是 confabulation (虚构)：模型写出看起来合理但做不到的代码。进化的天花板是搜索成本：对语法进行随机突变很少产生可编译的程序，更不用说更好的程序了。

AlphaEvolve（Novikov 等人，DeepMind，arXiv:2506.13131，2025 年 6 月）将两者结合。LLM 对程序数据库提出有针对性的编辑；自动评估器对每个变体打分；高分变体成为未来世代的父代。LLM 处理编写合理代码的昂贵步骤；评估器捕捉虚构内容。循环运行数小时到数周。

报告的结果：48 次标量乘法的 4x4 复数矩阵乘法（Strassen 1969 年的界限是 49），一个投入 Google 生产环境的 Borg 调度启发式算法，32.5% 的 FlashAttention 内核加速，Gemini 训练吞吐量改进。

该架构有效是因为评估器是机器可检查的。在评估器不是的地方它就不起作用。这种不对称性就是本课的要点。

## 概念

### 循环

1. 从一个正确但次优的种子程序 (seed program) `P_0` 开始。
2. 维护一个变体程序数据库，每个都由评估器打分。
3. 从数据库中采样一个或多个父代（MAP-elites 风格或基于岛屿的）。
4. 提示 LLM（Gemini Flash 用于大量候选，Gemini Pro 用于难的）生成父代的修改变体。
5. 在留出评估器上编译、运行和评估变体。
6. 根据其分数和特征向量插入数据库。
7. 重复。

两个细节很重要。首先，LLM 得到的提示不仅仅是父代程序——通常是数据库中的几个顶部变体，加上评估器签名，加上一个简短的任务描述。模型的任务是提出可能提高分数的有针对性变更。其次，数据库是结构化的（MAP-elites 网格，基于岛屿的），因此循环探索多样性，而不仅仅是当前领先者。

### 什么使评估器不可协商

AlphaEvolve 的胜利都来自评估器快速、确定性且难以博弈的领域：

- **矩阵乘法算法**：一个单元测试，乘以矩阵并按位检查相等性。
- **Borg 调度启发式算法**：一个生产级模拟器，重放历史集群负载并测量浪费的计算资源。
- **FlashAttention 内核**：一个正确性测试加上真实硬件上的 wall-clock 基准测试。
- **Gemini 训练吞吐量**：测量的每步 GPU 秒数。

在每种情况下，评估器都捕捉了否则会占主导地位的 LLM 错误类别：虚构的正确性声明、在硬件上消失的性能声明，以及边缘情况故障。移除评估器，循环就会优化漂亮的代码。

### 奖励黑客 (reward hacking) 是该陈述的另一面

进化会优化评估器测量的任何东西。如果评估器不完美，循环就会找到不完美之处。在未验证的领域中，循环会优化表面特征而非预期行为。DeepMind 在论文中明确指出了这一点：AlphaEvolve 的成功只转移到评估器严谨性与搜索野心相匹配的领域。

2025-2026 年代码搜索循环中奖励黑客的具体例子：

- 奖励"完成时间"的优化目标奖励提交空解。
- 奖励测试下正确性的基准分数奖励记忆测试和过拟合 (overfitting)。
- "代码质量"代理奖励移除注释和重命名变量，没有语义变化。

AlphaEvolve 中的修复：部署一个 LLM 从未见过的留出评估器，在评估时生成输入。即便如此，DeepMind 建议对任何提议的部署进行严格审查。

### 为什么 LLM + 搜索胜过单独任何一种

LLM 可以产生可编译的、语义上合理的修改。对 2000 行 Python 文件进行随机突变遗传算法几乎总是产生语法错误。LLM 还将搜索集中在合理的邻域（更改一个函数，而非随机字节），这大大减少了浪费的评估器调用。

评估器反过来捕捉 LLM 的虚构。LLM 会自信地声称一个函数"在极限情况下是 O(n log n)"，而实际上它是 O(n^2)；wall-clock 基准测试使这个问题尘埃落定。

### AlphaEvolve 在前沿技术栈中的位置

| 系统 | 生成器 | 评估器 | 领域 | 示例胜利 |
|---|---|---|---|---|
| AlphaEvolve | Gemini | 正确性 + 基准测试 | 算法、内核、调度器 | 48-mul 4x4 矩阵乘法 |
| FunSearch (DeepMind, 2023) | PaLM / Codey | 正确性 | 组合数学 | cap-set 下界 |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM 批评 + 实验 | ML 研究 | ICLR 研讨会论文 |
| Darwin Godel Machine (L4) | 智能体脚手架 | SWE-bench / Polyglot | 智能体代码 | 20% → 50% SWE-bench |

四种都是同一配方的变体：生成器加评估器，循环。区别在于评估器评分什么以及它有多严谨。

## 使用它

`code/main.py` 在一个玩具符号回归问题上实现了一个最小化的 AlphaEvolve 式循环。"LLM"是一个提出对计算目标函数的程序进行小语法突变的标准库代理。"评估器"测量留出测试点上的均方误差。

观察：

- 最佳分数如何随世代改善。
- MAP-elites 网格如何保持多样解存活，使循环不会收敛到局部最小值 (local minimum)。
- 移除留出测试（仅训练评估器）如何让循环 spectacularly 过拟合。

## 交付它

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑 AlphaEvolve 式循环的先决条件：你的评估器是否真的捕捉了你关心的失败？

## 练习

1. 运行 `code/main.py`。注意最佳分数轨迹。禁用留出评估器（标志 `--no-holdout`）并重新运行。量化过拟合。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites 网格的内容。为一个新问题（例如编译器优化遍）设计一个能保持搜索多样性的特征向量描述符。

3. 48 次乘法的 4x4 结果在 56 年后超越了 Strassen 的 49 次乘法界限。阅读论文附录 F，用三句话解释为什么这个问题的评估器特别容易做对，以及为什么大多数领域不是这样。

4. 提议一个 AlphaEvolve 会失败的领域。精确识别评估器在哪里失效以及为什么。

5. 为你知道的一个领域，写出你会使用的评估器签名。包括 (a) 正确性条件，(b) 性能指标，(c) 留出输入生成规则，(d) 至少一个反奖励黑客检查。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| AlphaEvolve | "DeepMind 的进化式编程智能体" | Gemini + 程序数据库 + 机器可检查评估器 |
| MAP-elites | "保持多样性的档案" | 以特征向量为键的网格；每个单元格持有该描述符的最佳变体 |
| Island model (岛屿模型) | "并行进化子种群" | 独立种群定期迁移；防止过早收敛 |
| Machine-checkable evaluator (机器可检查评估器) | "确定性预言机" | LLM 无法伪造的单元测试、模拟器或基准测试——该循环的先决条件 |
| Reward hacking (奖励黑客) | "优化测量指标而非目标" | 循环找到最大化分数而不执行预期任务的方法 |
| Seed program (种子程序) | "起点" | 循环从中进化的初始正确但次优的程序 |
| Held-out evaluator (留出评估器) | "LLM 从未见过的评估数据" | 在评估时生成的输入以防止记忆化 |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 厂商撰写的结果。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法，包括 48-mul 4x4 矩阵乘法。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将评估器约束的自主性框定为关键研究方向。
