---
name: translate-phase
description: Translate AI Engineering course materials (docs, code comments, quiz questions) from English to Chinese while preserving code executability and markdown structure. Use when user asks to translate a phase, lesson, chapter, or directory within phases/ into Chinese, or mentions translating course content, docs, or code comments.
---

# Translate Phase to Chinese

Translate course materials under `phases/` from English to Chinese.

## Scope

| File/Dir | Action |
|----------|--------|
| `docs/en.md` | Create `docs/zh.md` with full translation |
| `code/*.{py,jl,rs,js,ts,go}` | Translate comments and docstrings in-place |
| `quiz.json` | Translate question/option text in-place |
| `outputs/*.md` | Translate content, create `*-zh.md` or in-place |
| `README.md` | Translate to `README.zh.md` at phase root |

## Rules

### Code Comments
- Translate **only** comments (`#`, `//`, `/* */`, docstrings, string literals used as docs).
- **Never** translate: variable names, function names, class names, file names, keywords, API calls, non-doc strings.
- Preserve indentation and comment syntax.

### Technical Terms
- **Keep English terms as-is** in the main text.
- On **first occurrence** in each file, append Chinese in parentheses: `self-attention (自注意力)`.
- For subsequent occurrences in the same file, use English only.

| English | First-occurrence annotation |
|---------|----------------------------|
| self-attention | self-attention (自注意力) |
| multi-head attention | multi-head attention (多头注意力) |
| embedding | embedding (嵌入 / 词嵌入) |
| fine-tuning | fine-tuning (微调) |
| pre-training | pre-training (预训练) |
| inference | inference (推理) |
| gradient descent | gradient descent (梯度下降) |
| loss function | loss function (损失函数) |
| hyperparameter | hyperparameter (超参数) |
| batch size | batch size (批量大小) |
| epoch | epoch (轮次) |
| overfitting | overfitting (过拟合) |
| underfitting | underfitting (欠拟合) |
| regularization | regularization (正则化) |
| dropout | dropout (随机失活) |
| activation function | activation function (激活函数) |
| backpropagation | backpropagation (反向传播) |
| forward pass | forward pass (前向传播) |
| layer normalization | layer normalization (层归一化) |
| positional encoding | positional encoding (位置编码) |
| attention head | attention head (注意力头) |
| causal masking | causal masking (因果掩码) |
| softmax | softmax (软最大值) |

For terms not in this table: keep English, add the most widely accepted Chinese translation in parentheses on first use.

### Markdown & Structure
- Preserve all headings (`#`, `##`), lists, code blocks, tables, and math (`$...$`, `$$...$$`).
- Translate image alt text but not image paths.
- Keep frontmatter field names (like `title:`, `type:`) in English; translate their values.

### Quiz JSON
- Translate `question`, `options[]`, `explanation` fields.
- Apply the same "English first + Chinese annotation" rule to technical terms inside these fields.
- Leave `answer`, `id`, `type`, and code snippets in options unchanged.

## Workflow

1. Identify target phase/lesson path from user input.
2. List all translatable files in that scope.
3. Translate files in this order: `docs/en.md` → `code/` → `quiz.json` → `outputs/` → `README.md`.
4. After each file, do a quick sanity check: code still syntactically valid? Markdown structure intact?
5. Report what was translated and any edge cases encountered.

## Output

For each translated file, report:
- File path
- Word/line count (approximate)
- Any special handling applied (e.g., math preserved, code terms left untranslated)
