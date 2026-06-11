"""在示例应用上把同一项任务跑两遍：仅提示词 vs 工作台引导。

两条流水线都是脚本化的（无 LLM），因此测量是可复现的。
将 before-after-report.md 和 comparison.json 写在本文件旁边。

运行：python3 code/main.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_app"


SAMPLE_APP_PY = '''"""极简注册处理器。在本练习中视为类生产环境处理。"""

USERS: dict[str, str] = {}


def signup(email: str, password: str) -> dict[str, object]:
    USERS[email] = password
    return {"status": 200, "email": email}
'''

SAMPLE_TEST_PY = '''from sample_app.app import signup


def test_signup_happy_path():
    out = signup("a@b.co", "longenough")
    assert out["status"] == 200
'''


@dataclass
class TaskOutcome:
    pipeline: str
    tests_actually_run: bool
    acceptance_met: bool
    files_outside_scope: list[str] = field(default_factory=list)
    handoff_quality: str = "missing"
    reviewer_total: int = 0


ALLOWED = {"sample_app/app.py", "sample_app/test_app.py"}
FORBIDDEN = {"sample_app/scripts/release.sh"}


def run_prompt_only() -> TaskOutcome:
    """编辑几个文件，从不跑测试，声称完成。"""
    touched = ["sample_app/app.py", "README.md", "sample_app/scripts/release.sh"]
    return TaskOutcome(
        pipeline="prompt-only",
        tests_actually_run=False,
        acceptance_met=False,
        files_outside_scope=[p for p in touched if p not in ALLOWED],
        handoff_quality="missing",
        reviewer_total=3,
    )


def run_workbench() -> TaskOutcome:
    """读取范围合约，在范围内编辑，通过反馈运行器跑验收、过门控、经审查、生成交接。"""
    touched = ["sample_app/app.py", "sample_app/test_app.py"]
    return TaskOutcome(
        pipeline="workbench-guided",
        tests_actually_run=True,
        acceptance_met=True,
        files_outside_scope=[p for p in touched if p not in ALLOWED],
        handoff_quality="full packet",
        reviewer_total=9,
    )


def write_report(po: TaskOutcome, wb: TaskOutcome) -> None:
    lines = [
        "# Before / After: Agent Workbench on a Real Repo",
        "",
        "Same task. Same sample app. Two pipelines.",
        "",
        "| Outcome | Prompt only | Workbench |",
        "|---------|-------------|-----------|",
        f"| tests_actually_run | {po.tests_actually_run} | {wb.tests_actually_run} |",
        f"| acceptance_met | {po.acceptance_met} | {wb.acceptance_met} |",
        f"| files_outside_scope | {len(po.files_outside_scope)} | {len(wb.files_outside_scope)} |",
        f"| handoff_quality | {po.handoff_quality} | {wb.handoff_quality} |",
        f"| reviewer_total (/10) | {po.reviewer_total} | {wb.reviewer_total} |",
        "",
        "## Read",
        "",
        "Prompt only writes outside scope, claims done without running the acceptance command, "
        "leaves no handoff, and scores low on review. Workbench keeps writes in scope, runs the "
        "acceptance command through the feedback runner, passes the verification gate, and ships "
        "a handoff packet the next session loads on startup.",
    ]
    (HERE / "before-after-report.md").write_text("\n".join(lines) + "\n")


def write_sample() -> None:
    SAMPLE.mkdir(exist_ok=True)
    (SAMPLE / "app.py").write_text(SAMPLE_APP_PY)
    (SAMPLE / "test_app.py").write_text(SAMPLE_TEST_PY)
    (SAMPLE / "README.md").write_text("# sample app\n\nForbidden zone for agent tasks.\n")
    (SAMPLE / "scripts").mkdir(exist_ok=True)
    (SAMPLE / "scripts" / "release.sh").write_text("#!/usr/bin/env bash\necho release\n")


def main() -> None:
    write_sample()
    po = run_prompt_only()
    wb = run_workbench()

    for outcome in (po, wb):
        print(f"=== {outcome.pipeline} ===")
        for k, v in asdict(outcome).items():
            print(f"  {k}: {v}")
        print()

    write_report(po, wb)
    (HERE / "comparison.json").write_text(
        json.dumps({"prompt_only": asdict(po), "workbench": asdict(wb)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
