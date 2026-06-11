"""Self-hosted LLM engine decision-tree walker — stdlib Python.

自托管 LLM 引擎决策树遍历器 —— 仅使用 Python 标准库。

Given hardware, scale, and workload, pick an engine with explanation.
给定硬件、规模和工作负载，选择引擎并给出解释。
"""

from __future__ import annotations


def pick_engine(hardware: str, scale: str, workload: str) -> dict:
    reasons = []
    engine = None

    if hardware == "CPU":
        engine = "llama.cpp"
        reasons.append("hardware is CPU — only llama.cpp is competitive")
        reasons.append("硬件为 CPU —— 只有 llama.cpp 有竞争力")
        if scale == "single_user":
            reasons.append("single-user dev → Ollama wraps llama.cpp with one-command UX")
            reasons.append("单用户开发 → Ollama 封装 llama.cpp，一键体验")
            engine = "Ollama (llama.cpp under the hood)"
    elif hardware == "Apple Silicon":
        engine = "Ollama" if scale == "single_user" else "llama.cpp"
        reasons.append("Apple Silicon → Metal via llama.cpp (Ollama wraps)")
        reasons.append("Apple Silicon → 通过 llama.cpp 使用 Metal（Ollama 封装）")
    elif hardware == "AMD":
        engine = "vLLM"
        reasons.append("AMD → vLLM ROCm support; TRT-LLM is NVIDIA-only")
        reasons.append("AMD → vLLM 支持 ROCm；TRT-LLM 仅支持 NVIDIA")
        if "agentic" in workload.lower() or "prefix" in workload.lower():
            engine = "SGLang"
            reasons.append("agentic / prefix-heavy → SGLang RadixAttention")
            reasons.append("agentic / 前缀密集 → SGLang RadixAttention")
    elif hardware == "NVIDIA Hopper":
        if "agentic" in workload.lower() or "prefix" in workload.lower():
            engine = "SGLang"
            reasons.append("Hopper + agentic/prefix → SGLang is the specialist")
            reasons.append("Hopper + agentic/前缀密集 → SGLang 是专精")
        elif scale == "single_user":
            engine = "Ollama"
            reasons.append("single-user on Hopper is a dev scenario → Ollama is enough")
            reasons.append("Hopper 上单用户是开发场景 → Ollama 足够")
        else:
            engine = "vLLM"
            reasons.append("Hopper production → vLLM is the broad default")
            reasons.append("Hopper 生产环境 → vLLM 是广泛默认")
    elif hardware == "NVIDIA Blackwell":
        engine = "TRT-LLM"
        reasons.append("Blackwell + throughput priority → TRT-LLM leads on B200/GB200")
        reasons.append("Blackwell + 吞吐量优先 → TRT-LLM 在 B200/GB200 上领先")
        if scale in ("small_team", "production") and "agentic" not in workload.lower():
            reasons.append("vLLM Blackwell SM120 is a close second (v0.15.1 Feb 2026)")
            reasons.append("vLLM Blackwell SM120 是接近的第二名（v0.15.1 2026 年 2 月）")

    if scale == "enterprise":
        reasons.append("10k+ users → stack with production-stack (Phase 17 · 18)"
                      " + disaggregated (Phase 17 · 17) + cache-aware router (Phase 17 · 11)")
        reasons.append("10k+ 用户 → 叠加 production-stack（Phase 17 · 18）"
                      " + disaggregated（Phase 17 · 17）+ 缓存感知路由（Phase 17 · 11）")

    reasons.append("TGI is in maintenance mode since Dec 11, 2025 — default AWAY from TGI for new projects")
    reasons.append("TGI 自 2025 年 12 月 11 日起进入维护模式 —— 新项目默认远离 TGI")

    return {
        "hardware": hardware,
        "scale": scale,
        "workload": workload,
        "engine": engine,
        "reasons": reasons,
    }


SCENARIOS = [
    ("CPU",              "single_user",   "chat"),
    ("Apple Silicon",    "single_user",   "coding assistant"),
    ("NVIDIA Hopper",    "production",    "general chat"),
    ("NVIDIA Hopper",    "production",    "agentic multi-turn"),
    ("NVIDIA Blackwell", "enterprise",    "MoE frontier serving"),
    ("AMD",              "production",    "RAG with heavy prefix reuse"),
    ("NVIDIA Hopper",    "small_team",    "long-context 128K"),
]


def main() -> None:
    print("=" * 80)
    print("SELF-HOSTED ENGINE DECISION TREE — hardware / scale / workload")
    print("=" * 80)
    print("=" * 80)
    for hw, sc, wl in SCENARIOS:
        d = pick_engine(hw, sc, wl)
        print(f"\n[{hw}] [{sc}] [{wl}]")
        print(f"  → engine: {d['engine']}")
        for r in d["reasons"]:
            print(f"    · {r}")


if __name__ == "__main__":
    main()
