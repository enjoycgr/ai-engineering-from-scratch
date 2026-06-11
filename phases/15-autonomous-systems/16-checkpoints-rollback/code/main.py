"""带 checkpoint 的工作流，包含幂等性、前置条件、验证、回滚。

模拟四个场景：
  1. 干净运行
  2. 提交崩溃后重试  -> 幂等性防止双重执行
  3. 前置条件失败         -> 工作流中止而不触发
  4. 验证失败               -> 回滚触发
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass


# ---------- 迷你数据库 ----------

DB = {"balance_A": 1500, "balance_B": 200, "last_transfer_id": None}


def persist_transfer(txid: str, from_acct: str, to_acct: str, amount: int) -> None:
    DB[f"balance_{from_acct}"] -= amount
    DB[f"balance_{to_acct}"] += amount
    DB["last_transfer_id"] = txid


def rollback_transfer(txid: str, from_acct: str, to_acct: str, amount: int,
                      prior_last_transfer_id: str | None) -> None:
    # 补偿事务：恢复余额和先前的转账 id。
    DB[f"balance_{from_acct}"] += amount
    DB[f"balance_{to_acct}"] -= amount
    DB["last_transfer_id"] = prior_last_transfer_id


# ---------- Checkpoint 存储 ----------

@dataclass
class Checkpoint:
    path: str

    def __post_init__(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    def save(self, k: str, v: dict) -> None:
        # 原子写入：序列化到同级临时文件，fsync，然后
        # 重命名。如果进程在写入中途崩溃，原始文件
        # 仍然完整，因此下次重试找到先前的
        # 幂等性记录而非截断的 JSON 块。
        data = self.load()
        data[k] = v
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)


# ---------- Workflow ----------

def key(txid: str) -> str:
    return hashlib.sha256(txid.encode()).hexdigest()[:12]


def run_transfer(cp: Checkpoint, txid: str, from_acct: str, to_acct: str,
                 amount: int, min_balance: int,
                 inject_crash_after_execute: bool = False,
                 inject_verify_fail: bool = False) -> str:
    k = key(txid)
    record = cp.load().get(k, {"status": "new"})

    # 所有终态间的幂等性。对相同 txid 的重试
    # 在任何终态判决后 —— committed, verified, rolled-back,
    # aborted-precondition —— 必须短路到原始结果
    # 而非重新执行。
    terminal_results = {
        "committed": "idempotent-skip",
        "verified": "ok",
        "rolled-back": "verify-fail-rolled-back",
        "aborted-precondition": "aborted-precondition",
    }
    if record["status"] in terminal_results:
        return terminal_results[record["status"]]

    # 前置条件检查：转账后余额必须保持 >= min_balance
    if DB[f"balance_{from_acct}"] - amount < min_balance:
        cp.save(k, {"status": "aborted-precondition", "txid": txid})
        return "aborted-precondition"

    # 捕获先前状态以便回滚可以精确恢复（而非仅仅反转）。
    prior_last_transfer_id = DB["last_transfer_id"]

    # 在副作用前记录意图，这样在 save 和 persist_transfer 之间崩溃
    # 会留下一个 "committed" 标记，重试可以检测并短路。
    # 只有当下方的 action 后读取确认副作用已落地时，
    # 我们才提升到 "verified"。
    #
    # 微妙的持久性缺口（课程权衡）：如果进程在 cp.save 之后
    # 和 persist_transfer 之前崩溃，重试会看到
    # status == "committed" 并返回 "idempotent-skip"，尽管
    # 转账从未真正运行。生产系统通过以下方式关闭此缺口：
    # (a) 将幂等性键带入副作用本身，使目标 DB 强制执行 exactly-once，
    # 或 (b) 在目标的后 action 读取上把关 "committed"，
    # 这正是下方 verify 步骤对非崩溃路径所做的。
    cp.save(k, {"status": "committed", "txid": txid,
                "from_acct": from_acct, "to_acct": to_acct,
                "amount": amount,
                "prior_last_transfer_id": prior_last_transfer_id})
    persist_transfer(txid, from_acct, to_acct, amount)
    if inject_crash_after_execute:
        raise RuntimeError("simulated crash after execute")

    # 动作后验证
    if inject_verify_fail or DB["last_transfer_id"] != txid:
        rollback_transfer(txid, from_acct, to_acct, amount, prior_last_transfer_id)
        cp.save(k, {"status": "rolled-back", "txid": txid})
        return "verify-fail-rolled-back"

    cp.save(k, {"status": "verified", "txid": txid})
    return "ok"


# ---------- Driver ----------

def main() -> None:
    print("=" * 80)
    print("CHECKPOINTS AND ROLLBACK (Phase 15, Lesson 16)")
    print("=" * 80)

    tmp = tempfile.mkdtemp()
    print()
    print("场景 1：干净运行")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp1.json"))
    out = run_transfer(cp, "tx-001", "A", "B", 100, min_balance=200)
    print(f"  result={out}  DB={DB}")

    print("\nScenario 2: crash mid-commit, retry (idempotency catches)")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp2.json"))
    try:
        run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200,
                     inject_crash_after_execute=True)
    except RuntimeError as e:
        print(f"  crash: {e}")
    # 崩溃后重试
    out = run_transfer(cp, "tx-002", "A", "B", 100, min_balance=200)
    print(f"  retry result={out}  DB={DB}")

    print("\n场景 3：前置条件失败（余额会低于最小值）")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp3.json"))
    out = run_transfer(cp, "tx-003", "A", "B", 10_000, min_balance=200)
    print(f"  result={out}  DB={DB}")

    print("\n场景 4：验证失败 -> 回滚")
    print("-" * 80)
    cp = Checkpoint(os.path.join(tmp, "cp4.json"))
    balances_before = dict(DB)
    out = run_transfer(cp, "tx-004", "A", "B", 100, min_balance=200,
                       inject_verify_fail=True)
    balances_after = dict(DB)
    print(f"  result={out}  balances_before_after_equal="
          f"{balances_before == balances_after}")

    print()
    print("=" * 80)
    print("HEADLINE: 幂等性 + 前置条件 + 验证 + 回滚")
    print("-" * 80)
    print("  四个部分，而非一个。每个覆盖不同的失败类别：")
    print("  幂等性 -> 崩溃时重试安全")
    print("  前置条件 -> 审批与提交之间的状态漂移")
    print("  验证       -> 我们认为发生时副作用并未发生")
    print("  回滚     -> 已知不良状态恢复或告警")
    print("  Article 14 操作解读：checkpoint 可查询，回滚")
    print("  已演练，审计追踪在部署中存活。


if __name__ == "__main__":
    main()
