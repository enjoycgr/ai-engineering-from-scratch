"""四个 multi-agent 原语，仅使用标准库。

原语：
  - Agent(name, system_prompt, tools, policy)
  - Handoff(from_agent, to_agent, reason)
  - SharedState（线程安全的消息池）
  - Orchestrator（Static、Handoff-driven、LLM-selected）

在三种 orchestrator 类型下运行相同的三 agent pipeline（researcher -> writer -> reviewer）。
Agent 是 scripted policies，不是 LLM 调用——重点是协调结构。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


Message = dict


@dataclass
class SharedState:
    messages: list[Message] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, msg: Message) -> None:
        with self._lock:
            self.messages.append(msg)

    def snapshot(self) -> list[Message]:
        with self._lock:
            return list(self.messages)

    def last_by(self, name: str) -> Optional[Message]:
        with self._lock:
            for m in reversed(self.messages):
                if m["from"] == name:
                    return m
            return None


@dataclass
class Agent:
    name: str
    system_prompt: str
    policy: Callable[[SharedState], Message]

    def run(self, state: SharedState) -> Message:
        msg = self.policy(state)
        msg.setdefault("from", self.name)
        return msg


def researcher_policy(state: SharedState) -> Message:
    n = len([m for m in state.snapshot() if m["from"] == "researcher"])
    notes = f"note {n + 1}: FIPA-ACL ratified 2000; 20 performatives."
    return {"content": notes, "handoff": "writer" if n == 0 else "done"}


def writer_policy(state: SharedState) -> Message:
    research = [m["content"] for m in state.snapshot() if m["from"] == "researcher"]
    draft = "Draft summarizing: " + " | ".join(research) if research else "Draft with no research yet."
    return {"content": draft, "handoff": "reviewer"}


def reviewer_policy(state: SharedState) -> Message:
    last = state.last_by("writer")
    verdict = "approved" if last and "summarizing" in last["content"] else "needs revision"
    return {"content": f"Review verdict: {verdict}.", "handoff": "done"}


def make_team() -> dict[str, Agent]:
    return {
        "researcher": Agent("researcher", "Gather facts.", researcher_policy),
        "writer": Agent("writer", "Draft from research.", writer_policy),
        "reviewer": Agent("reviewer", "Critique the draft.", reviewer_policy),
    }


class StaticOrchestrator:
    """固定的顺序执行，LangGraph 风格的确定性边。"""

    def __init__(self, order: list[str]) -> None:
        self.order = order

    def run(self, team: dict[str, Agent], state: SharedState, max_steps: int = 10) -> None:
        for name in self.order[:max_steps]:
            msg = team[name].run(state)
            state.append(msg)


class HandoffOrchestrator:
    """OpenAI Swarm 风格：当前 agent 返回自己的 handoff 目标。"""

    def __init__(self, start: str) -> None:
        self.start = start

    def run(self, team: dict[str, Agent], state: SharedState, max_steps: int = 10) -> None:
        current = self.start
        for _ in range(max_steps):
            if current not in team:
                return
            msg = team[current].run(state)
            state.append(msg)
            nxt = msg.get("handoff", "done")
            if nxt == "done":
                return
            current = nxt


class LLMSelectorOrchestrator:
    """AutoGen GroupChat 风格的 speaker selection。这里的 selector function
    是脚本化的，但在生产环境中它会是一个读取消息池的 LLM 调用。"""

    def __init__(self, start: str, selector: Callable[[SharedState, dict[str, Agent]], Optional[str]]) -> None:
        self.start = start
        self.selector = selector

    def run(self, team: dict[str, Agent], state: SharedState, max_steps: int = 10) -> None:
        current: Optional[str] = self.start
        for _ in range(max_steps):
            if current is None or current not in team:
                return
            msg = team[current].run(state)
            state.append(msg)
            current = self.selector(state, team)


def round_robin_selector(state: SharedState, team: dict[str, Agent]) -> Optional[str]:
    if not state.messages:
        return None
    last = state.messages[-1]["from"]
    names = list(team.keys())
    idx = (names.index(last) + 1) % len(names)
    if len([m for m in state.messages if m["from"] == "reviewer"]) >= 1:
        return None
    return names[idx]


def render_pool(label: str, state: SharedState) -> None:
    print(f"\n=== {label} ===")
    for i, m in enumerate(state.snapshot()):
        ho = f" -> {m['handoff']}" if "handoff" in m else ""
        print(f"  [{i}] {m['from']:10s} | {m['content']}{ho}")


def main() -> None:
    print("Four multi-agent primitives demo")
    print("-" * 42)

    team = make_team()
    state_a = SharedState()
    StaticOrchestrator(["researcher", "writer", "reviewer"]).run(team, state_a)
    render_pool("Static (LangGraph-style)", state_a)

    team = make_team()
    state_b = SharedState()
    HandoffOrchestrator("researcher").run(team, state_b)
    render_pool("Handoff-driven (OpenAI Swarm-style)", state_b)

    team = make_team()
    state_c = SharedState()
    LLMSelectorOrchestrator("researcher", round_robin_selector).run(team, state_c)
    render_pool("LLM-selected (AutoGen-style)", state_c)

    print("\nTakeaway: agents and state are identical across runs;")
    print("only the orchestrator choice changes who speaks when.")


if __name__ == "__main__":
    main()
