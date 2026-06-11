"""Propose-then-commit HITL 状态机 —— stdlib Python。

四个阶段：
  1. propose：智能体用幂等性键持久化拟议动作
  2. surface：评审员看到元数据（intent, lineage, blast, rollback）
  3. commit：需要正向确认；幂等
  4. verify：提交后重新读取目标资源

三个演示：
  - 干净的审批流
  - 瞬态失败后重试 -> 幂等性捕获
  - rubber-stamp UI vs challenge-and-response 清单
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class Proposal:
    thread_id: str
    action: str
    payload: dict
    intent: str
    lineage: str
    blast_radius: str
    rollback: str

    def key(self) -> str:
        sig = json.dumps({"t": self.thread_id, "a": self.action,
                          "p": self.payload}, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]


@dataclass
class Store:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def all(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, key: str, record: dict) -> None:
        data = self.all()
        data[key] = record
        with open(self.path, "w") as f:
            json.dump(data, f)


# ---------- 已执行副作用追踪器（假装是后端） ----------

SIDE_EFFECTS: list[str] = []


def execute(proposal: Proposal) -> bool:
    SIDE_EFFECTS.append(f"{proposal.action}:{json.dumps(proposal.payload)}")
    return True


def verify(proposal: Proposal) -> bool:
    # 在真实系统中，这会重新读取目标资源。
    needle = f"{proposal.action}:{json.dumps(proposal.payload)}"
    return needle in SIDE_EFFECTS


# ---------- 流程 ----------

def propose(store: Store, p: Proposal) -> str:
    k = p.key()
    existing = store.all().get(k)
    if existing:
        print(f"  [propose] idempotent: record {k} already exists "
              f"(status={existing['status']})")
        return k
    record = {"status": "waiting", **vars(p)}
    store.save(k, record)
    print(f"  [propose] record {k} stored, waiting for review")
    return k


def surface(store: Store, k: str) -> None:
    r = store.all()[k]
    print(f"  [surface] proposal {k}")
    # 使用 'name' 而非 'field' 以避免遮蔽 dataclasses.field，
    # 以防读者后续在此模块下方添加 dataclass (Ruff F402)。
    for name in ("intent", "lineage", "blast_radius", "rollback"):
        print(f"    {name:<14} {r[name]}")


def rubber_stamp_approve(store: Store, k: str) -> bool:
    r = store.all()
    rec = r[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "rubber_stamp"
    store.save(k, rec)
    print("  [approve:rubber-stamp] clicked Approve (no checklist)")
    return True


def checklist_approve(store: Store, k: str,
                      understood: bool, verified: bool,
                      rollback_ready: bool) -> bool:
    if not (understood and verified and rollback_ready):
        print("  [approve:checklist] REJECTED (incomplete answers)")
        return False
    r = store.all()
    rec = r[k]
    rec["status"] = "approved"
    rec["ack_mode"] = "challenge_response"
    store.save(k, rec)
    print("  [approve:checklist] APPROVED (all three checks)")
    return True


def commit(store: Store, k: str) -> bool:
    data = store.all()
    rec = data[k]
    if rec["status"] == "committed":
        print(f"  [commit] idempotent: {k} already committed, no re-execute")
        return True
    if rec["status"] != "approved":
        print(f"  [commit] refusing: {k} status={rec['status']}")
        return False
    p = Proposal(
        thread_id=rec["thread_id"], action=rec["action"],
        payload=rec["payload"], intent=rec["intent"],
        lineage=rec["lineage"], blast_radius=rec["blast_radius"],
        rollback=rec["rollback"],
    )
    execute(p)
    rec["status"] = "committed"
    store.save(k, rec)
    print(f"  [commit] executed; verify={verify(p)}")
    return True


# ---------- 演示 ----------

def main() -> None:
    print("=" * 80)
    print("PROPOSE-THEN-COMMIT HITL (Phase 15, Lesson 15)")
    print("=" * 80)
    tmp = tempfile.mkdtemp()
    store = Store(os.path.join(tmp, "proposals.json"))

    p = Proposal(
        thread_id="t-001",
        action="email.send",
        payload={"to": "team@example.com", "subject": "release"},
        intent="Announce the v1.2 release to the team list",
        lineage="Release notes page /releases/1.2",
        blast_radius="37 recipients; wrong send = external embarrassment",
        rollback="no in-band rollback; follow up with correction email",
    )

    print("\n演示 1：干净审批流（challenge-and-response）")
    print("-" * 80)
    k = propose(store, p)
    surface(store, k)
    checklist_approve(store, k, understood=True, verified=True, rollback_ready=True)
    commit(store, k)

    print("\n演示 2：审批后重试；幂等性捕获重复执行")
    print("-" * 80)
    initial = len(SIDE_EFFECTS)
    commit(store, k)  # retry
    commit(store, k)  # retry
    print(f"  2 次重试后总副作用数: {len(SIDE_EFFECTS)} "
          f"(原 {initial}) -> idempotent")

    print("\n演示 3：rubber-stamp UI vs challenge-and-response")
    print("-" * 80)
    p2 = Proposal(
        thread_id="t-002", action="db.update",
        payload={"row": 42, "col": "status", "val": "closed"},
        intent="Close a stale issue",
        lineage="periodic scan of stale-issue dashboard",
        blast_radius="one DB row; reversible within 1h backup window",
        rollback="restore row from nightly backup",
    )
    k2 = propose(store, p2)
    rubber_stamp_approve(store, k2)
    commit(store, k2)

    p3 = Proposal(
        thread_id="t-003", action="db.drop_table",
        payload={"table": "old_users"},
        intent="Drop an unused table (per cleanup runbook)",
        lineage="runbook #RB-17",
        blast_radius="destructive; 420k rows dropped; not reversible within 24h",
        rollback="restore from weekly backup; data loss up to 6 days",
    )
    k3 = propose(store, p3)
    # Reviewer cannot tick rollback-ready; checklist_approve declines
    ok = checklist_approve(store, k3, understood=True, verified=True,
                           rollback_ready=False)
    # 教学意图：对被拒绝的提案调用 commit()，使得
    # 日志展示 commit() 在状态仍为 "waiting" 而非 "approved" 时拒绝。
    # 我们希望拒绝行被打印出来。
    if not ok:
        commit(store, k3)

    print()
    print("=" * 80)
    print("HEADLINE: 让结构化评审成为阻力最小的路径")
    print("-" * 80)
    print("  幂等性键防止重试时的双重执行。")
    print("  持久性让审批迟到两天仍然适用。")
    print("  Challenge-and-response 清单是 rubber-stamp 审批的")
    print("  文档化缓解措施；EU AI Act Article 14 期望它。")
    print("  提交后验证消除 '以为它发生了' 的类别。")


if __name__ == "__main__":
    main()
