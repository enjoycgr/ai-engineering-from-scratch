#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

MODEL = "qwen3.5:27b"

PROMPT = """Translate the following course file to Simplified Chinese.

Rules:
- Output ONLY the translated file content. No explanations, no markdown wrapper.
- Preserve Markdown structure, fenced code blocks, Mermaid, math formulas, tables, and frontmatter keys.
- Do not translate code block contents unless they are user-facing natural-language examples.
- Keep major technical terms in English; on first occurrence in this file append Chinese parentheses, e.g. instance segmentation (实例分割), GAN (生成对抗网络), diffusion (扩散), denoising (去噪), latent space (潜空间), Vision Transformer (视觉 Transformer), self-attention (自注意力), embedding (嵌入).
- After first occurrence in the same file, the English term alone is acceptable.
- Do not create half-translated English/Chinese sentences; prose sentences should be natural Simplified Chinese.
- Keep file paths, identifiers, API names, commands, package names, and code snippets unchanged.

FILE CONTENT START
{content}
FILE CONTENT END
"""

def clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text + "\n"

def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    content = src.read_text()
    prompt = PROMPT.format(content=content)
    proc = subprocess.run(["ollama", "run", MODEL], input=prompt, text=True, capture_output=True, check=True)
    dst.write_text(clean(proc.stdout))

if __name__ == "__main__":
    main()
