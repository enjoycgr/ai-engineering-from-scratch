"""极简 durable-execution 引擎 —— stdlib Python。

模拟 Temporal、LangGraph checkpointing、Microsoft Agent Framework
和 Claude Code Routines 使用的 workflow / activity / event-log 模式。

活动在执行前记录输入，执行后记录输出。工作流的重放
重新运行工作流代码，但返回已缓存的输出
给日志中已有事件的活动。运行中崩溃仅丢失
未完成的活动。
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from dataclasses import dataclass


# ---------- 事件日志 ----------

@dataclass
class EventLog:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def events(self) -> list[dict]:
        with open(self.path) as f:
            return json.load(f)

    def append(self, ev: dict) -> None:
        evs = self.events()
        evs.append(ev)
        with open(self.path, "w") as f:
            json.dump(evs, f)

    def lookup(self, name: str, args: tuple) -> dict | None:
        for ev in self.events():
            if ev["name"] == name and ev["args"] == list(args) and ev["status"] == "done":
                return ev
        return None


# ---------- Activity 装饰器 ----------

def activity(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(log: EventLog, *args):
            hit = log.lookup(name, args)
            if hit:
                print(f"    [replay] {name}({args}) -> {hit['result']} (from log)")
                return hit["result"]
            log.append({"name": name, "args": list(args), "status": "started"})
            result = fn(*args)
            log.append({"name": name, "args": list(args),
                        "status": "done", "result": result})
            print(f"    [run]    {name}({args}) -> {result}")
            return result
        return wrapper
    return deco


# ---------- 示例 activities ----------

@activity("fetch_docs")
def fetch_docs(query: str) -> int:
    # 假装调用 API；返回文档数。
    return len(query) * 3


@activity("call_llm")
def call_llm(doc_count: int) -> str:
    # 假装 LLM 调用；此处为教学目的而确定性。
    return f"summary({doc_count}_docs)"


@activity("write_report")
def write_report(summary: str) -> str:
    # 假装有副作用的工具调用。
    return f"report://{summary}"


# ---------- 工作流 ----------

def workflow(log: EventLog, query: str, crash_after: int = -1) -> str:
    """三个 activity 的工作流，可选崩溃用于教学。"""
    doc_count = fetch_docs(log, query)
    if crash_after == 1:
        raise RuntimeError("simulated crash after fetch_docs")
    summary = call_llm(log, doc_count)
    if crash_after == 2:
        raise RuntimeError("simulated crash after call_llm")
    report = write_report(log, summary)
    return report


# ---------- 驱动 ----------

def reset_log(path: str) -> EventLog:
    if os.path.exists(path):
        os.remove(path)
    return EventLog(path)


def count_runs(log: EventLog) -> int:
    return sum(1 for ev in log.events() if ev["status"] == "started")


def main() -> None:
    print("=" * 70)
    print("DURABLE EXECUTION (Phase 15, Lesson 12)")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp()

    # 天真重试：崩溃时丢失事件日志。每次重启都重新运行
    # 所有内容。
    print("\nNaive retry (no event log persisted)")
    print("-" * 70)
    for attempt in range(1, 4):
        log = reset_log(os.path.join(tmpdir, "naive.json"))
        print(f"  attempt {attempt}:")
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> result {r}")
            print(f"    -> {count_runs(log)} activity starts this attempt")
            break
        except RuntimeError as e:
            print(f"    -> crash: {e}; {count_runs(log)} activity starts wasted")

    # 持久重试：跨尝试保持事件日志；重放不会
    # 重新执行已完成的活动。
    print("\nDurable retry (event log preserved across attempts)")
    print("-" * 70)
    durable_path = os.path.join(tmpdir, "durable.json")
    if os.path.exists(durable_path):
        os.remove(durable_path)

    for attempt in range(1, 4):
        log = EventLog(durable_path)
        print(f"  attempt {attempt}:")
        try:
            crash = 2 if attempt == 1 else -1
            r = workflow(log, "hello", crash_after=crash)
            print(f"    -> result {r}")
            print(f"    -> {count_runs(log)} total activity starts across attempts")
            break
        except RuntimeError as e:
            print(f"    -> crash: {e}")

    print()
    print("=" * 70)
    print("HEADLINE: 持久性使长时程运行的失败变得可承受")
    print("-" * 70)
    print("  天真重试在每次尝试时都重新执行每个活动。")
    print("  持久重试从日志中重放已完成的活动；")
    print("  只有缺失的活动真正运行。Temporal、LangGraph checkpointing、")
    print("  Microsoft Agent Framework 和 Claude Code Routines 使用相同设计。")
    print("  LLM 调用只是日志中另一个非确定性活动。")


if __name__ == "__main__":
    main()
