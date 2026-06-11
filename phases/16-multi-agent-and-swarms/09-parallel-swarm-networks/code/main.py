"""Swarm architecture (蜂群架构) demo: worker 从共享队列拉取任务。

在可变持续时间工作负载上对比三种调度策略：
  - sequential (顺序) (1 个 worker 处理所有任务)
  - fixed assignment (固定分配) (每个任务预分配给特定 worker)
  - swarm (4 个 worker 从共享队列拉取)

Swarm 自动平衡负载；固定分配会让快 worker 空闲。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    duration: float
    pre_assigned: int  # for the fixed-assignment baseline


def fake_work(task: Task) -> str:
    time.sleep(task.duration)
    return f"task-{task.task_id}-done"


def run_sequential(tasks: list[Task]) -> tuple[float, dict[int, int]]:
    t0 = time.time()
    counts: dict[int, int] = {0: 0}
    for t in tasks:
        fake_work(t)
        counts[0] += 1
    return time.time() - t0, counts


def run_fixed_assignment(tasks: list[Task], n_workers: int) -> tuple[float, dict[int, int]]:
    """每个任务预分配给 worker id。Worker 串行处理其任务。"""
    per_worker: dict[int, list[Task]] = {i: [] for i in range(n_workers)}
    for t in tasks:
        per_worker[t.pre_assigned].append(t)
    counts: dict[int, int] = {i: 0 for i in range(n_workers)}

    def worker(wid: int) -> None:
        for t in per_worker[wid]:
            fake_work(t)
            counts[wid] += 1

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.time() - t0, counts


def run_swarm(tasks: list[Task], n_workers: int) -> tuple[float, dict[int, int]]:
    """Worker 从共享队列拉取。"""
    q: queue.Queue = queue.Queue()
    for t in tasks:
        q.put(t)
    counts: dict[int, int] = {i: 0 for i in range(n_workers)}
    lock = threading.Lock()

    def worker(wid: int) -> None:
        while True:
            try:
                task = q.get_nowait()
            except queue.Empty:
                return
            fake_work(task)
            with lock:
                counts[wid] += 1
            q.task_done()

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return time.time() - t0, counts


def make_tasks(n_workers: int = 4) -> list[Task]:
    """8 个任务: 一半快 (0.1s), 一半慢 (0.4s)。Pre-assignment 是悲观的:
    worker 0 获得所有慢任务, 其他 worker 获得快任务。"""
    tasks: list[Task] = []
    for i in range(8):
        is_slow = i < 4
        tasks.append(
            Task(
                task_id=i,
                duration=0.4 if is_slow else 0.1,
                pre_assigned=0 if is_slow else (i - 3) % n_workers,
            )
        )
    return tasks


def main() -> None:
    print("Swarm architecture demo — 可变持续时间工作负载")
    print("-" * 56)
    n_workers = 4

    tasks = make_tasks(n_workers)
    total_work = sum(t.duration for t in tasks)
    print(f"{len(tasks)} 个任务, 4 个慢 (0.4s) + 4 个快 (0.1s)")
    print(f"总工作量-秒: {total_work:.2f}s")
    print(f"理想并行时间 ({n_workers} 个 worker): {total_work / n_workers:.2f}s")

    seq_time, seq_counts = run_sequential(tasks)
    print(f"\nSequential (1 个 worker):      wall={seq_time:.2f}s, counts={seq_counts}")

    fixed_time, fixed_counts = run_fixed_assignment(tasks, n_workers)
    print(f"Fixed assignment ({n_workers} 个 worker): wall={fixed_time:.2f}s, counts={fixed_counts}")
    print("  worker 0 获得所有 4 个慢任务; 其他 worker 在快任务完成后空闲。")

    swarm_time, swarm_counts = run_swarm(tasks, n_workers)
    print(f"Swarm ({n_workers} 个 worker):            wall={swarm_time:.2f}s, counts={swarm_counts}")
    print("  自动负载均衡 — 慢 worker 先完成, 快的拉取下一个作业。")

    speedup_vs_seq = seq_time / swarm_time if swarm_time > 0 else float("inf")
    speedup_vs_fixed = fixed_time / swarm_time if swarm_time > 0 else float("inf")
    print(f"\nSwarm 相对 sequential 加速: {speedup_vs_seq:.2f}x")
    print(f"Swarm 相对 fixed 加速:      {speedup_vs_fixed:.2f}x")
    print("\n要点: 当持续时间变化且分配难以预测时, swarm 获胜。")
    print("权衡: 没有中央 trace; 调试需要每个任务的 ID 和持久化日志。")


if __name__ == "__main__":
    main()
