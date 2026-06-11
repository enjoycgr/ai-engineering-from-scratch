"""Shared memory patterns: MessagePool, Blackboard, and a poisoning demo.

运行一个三 agent 研究任务两次。第一次运行有一个幻觉的
小数点，通过 shared memory (共享内存) 传播到最终报告。第二次
运行添加了一个 read-only verifier (只读验证器)，重新获取来源并标记
不一致。
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProvenanceEntry:
    id: int
    writer: str
    topic: str
    content: str
    timestamp: float
    prompt_hash: str
    source_uri: str | None = None
    supersedes: int | None = None
    flags: list[str] = field(default_factory=list)


class MessagePool:
    """Append-only full-pool shared state (仅追加的完整池共享状态)。"""

    def __init__(self) -> None:
        self.entries: list[ProvenanceEntry] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def write(self, writer: str, content: str, prompt: str, source_uri: str | None = None,
              topic: str = "default", supersedes: int | None = None) -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(
                id=eid,
                writer=writer,
                topic=topic,
                content=content,
                timestamp=time.time(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                source_uri=source_uri,
                supersedes=supersedes,
            )
            self.entries.append(e)
            return eid

    def read_all(self) -> list[ProvenanceEntry]:
        with self._lock:
            return list(self.entries)

    def flag(self, entry_id: int, flag: str) -> None:
        with self._lock:
            for e in self.entries:
                if e.id == entry_id:
                    e.flags.append(flag)
                    return


class Blackboard:
    """Topic-keyed pub/sub blackboard (按主题的发布/订阅黑板)。"""

    def __init__(self) -> None:
        self.topics: dict[str, list[ProvenanceEntry]] = {}
        self.subscribers: dict[str, list[Callable[[ProvenanceEntry], None]]] = {}
        self._lock = threading.Lock()
        self._next_id = 0

    def publish(self, writer: str, topic: str, content: str, prompt: str,
                source_uri: str | None = None) -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            e = ProvenanceEntry(
                id=eid,
                writer=writer,
                topic=topic,
                content=content,
                timestamp=time.time(),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:10],
                source_uri=source_uri,
            )
            self.topics.setdefault(topic, []).append(e)
            subs = list(self.subscribers.get(topic, []))
        for cb in subs:
            cb(e)
        return eid

    def subscribe(self, topic: str, cb: Callable[[ProvenanceEntry], None]) -> None:
        with self._lock:
            self.subscribers.setdefault(topic, []).append(cb)

    def read_topic(self, topic: str) -> list[ProvenanceEntry]:
        with self._lock:
            return list(self.topics.get(topic, []))


FAKE_SOURCES = {
    "https://arxiv.org/paper-1": "The study reports a 4.2% accuracy improvement over the baseline.",
    "https://arxiv.org/paper-2": "Dataset size was 12,500 examples.",
}


def retrieval_agent(pool: MessagePool, uri: str, hallucinate: bool) -> int:
    content = FAKE_SOURCES[uri]
    if hallucinate and "4.2%" in content:
        content = content.replace("4.2%", "42%")
    return pool.write(
        writer="retriever",
        content=content,
        prompt=f"Fetch and summarize {uri}",
        source_uri=uri,
    )


def summarizer_agent(pool: MessagePool) -> int:
    retrieved = [e for e in pool.read_all() if e.writer == "retriever"]
    if not retrieved:
        return pool.write("summarizer", "no source", "Summarize retrieval", None)
    latest = retrieved[-1].content
    summary = f"Summary: study reports a significant result -- {latest.split('.')[0]}."
    return pool.write("summarizer", summary, "Summarize retrieval", None)


def analyst_agent(pool: MessagePool) -> int:
    summaries = [e for e in pool.read_all() if e.writer == "summarizer"]
    if not summaries:
        return pool.write("analyst", "no summary", "Draw conclusions", None)
    latest = summaries[-1].content
    verdict = "Recommend adoption" if "42%" in latest else "Recommend further review"
    return pool.write("analyst", f"Analyst verdict: {verdict} (based on: {latest})",
                      "Draw conclusions", None)


def verifier_agent(pool: MessagePool) -> list[tuple[int, str]]:
    """Read-only agent (只读智能体)。重新获取引用的来源并标记不一致。

    返回一个 (entry_id, reason) 元组列表，供调用方处理。
    Verifier 从不写回 pool —— 由调用方决定如何处理。
    """
    findings = []
    for e in pool.read_all():
        if e.source_uri and e.source_uri in FAKE_SOURCES:
            truth = FAKE_SOURCES[e.source_uri]
            if e.content != truth:
                findings.append((e.id, f"mismatch with {e.source_uri}: fetched text was {truth!r}"))
    return findings


def run_without_verifier() -> None:
    print("=" * 72)
    print("RUN 1 — 无 verifier；幻觉传播")
    print("=" * 72)
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    analyst_agent(pool)
    for e in pool.read_all():
        print(f"  [{e.id}] {e.writer:11s} ({e.prompt_hash}) :: {e.content}")
    print("\n最终报告使用了幻觉的 42% 数字；没有触发警报。")


def run_with_verifier() -> None:
    print("\n" + "=" * 72)
    print("RUN 2 — read-only verifier 重新获取来源并标记")
    print("=" * 72)
    pool = MessagePool()
    retrieval_agent(pool, "https://arxiv.org/paper-1", hallucinate=True)
    summarizer_agent(pool)
    findings = verifier_agent(pool)
    for eid, reason in findings:
        pool.flag(eid, reason)
    analyst_agent(pool)

    for e in pool.read_all():
        flag_str = f" [FLAGGED: {'; '.join(e.flags)}]" if e.flags else ""
        print(f"  [{e.id}] {e.writer:11s} ({e.prompt_hash}) :: {e.content}{flag_str}")
    if findings:
        print(f"\nverifier 发现了 {len(findings)} 处不一致。下游 agent 可以抑制该结论。")


def demo_blackboard() -> None:
    print("\n" + "=" * 72)
    print("BLACKBOARD DEMO — 按 topic 的 pub/sub，不是每个 agent 都读取一切")
    print("=" * 72)
    bb = Blackboard()
    received = {"prices": [], "alerts": []}

    def on_prices(e: ProvenanceEntry) -> None:
        received["prices"].append(e.id)

    def on_alerts(e: ProvenanceEntry) -> None:
        received["alerts"].append(e.id)

    bb.subscribe("prices", on_prices)
    bb.subscribe("alerts", on_alerts)

    bb.publish("scraper-1", "prices", "AAPL=192.4", "poll market")
    bb.publish("scraper-2", "prices", "MSFT=401.2", "poll market")
    bb.publish("risk-engine", "alerts", "ALERT: AAPL moved >2% in 60s", "watch prices")

    print(f"  price subscribers got ids: {received['prices']}")
    print(f"  alert subscribers got ids: {received['alerts']}")
    print("  (注意：price subscribers 从未看到 alert；这就是重点)")


def main() -> None:
    run_without_verifier()
    run_with_verifier()
    demo_blackboard()
    print("\n要点：")
    print("  1. 没有 provenance 的 shared state 会将幻觉洗白进下游推理")
    print("  2. 具有独立来源访问权限的 read-only verifier 能捕获 memory poisoning")
    print("  3. blackboard 比 full pool 更具扩展性，因为 agent 只读取它们订阅的内容")


if __name__ == "__main__":
    main()
