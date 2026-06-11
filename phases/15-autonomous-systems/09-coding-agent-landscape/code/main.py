"""CodeAct vs JSON tool-call scaffold 对比 —— stdlib Python。

两种 scaffold 使用相同的存根 "模型"（确定性规则），因此
对比将 scaffold 与模型质量隔离。指标：
  - 解决的任务数
  - 使用的轮数
  - 每动作爆炸半径 (blast radius)（一个动作可触及的文件数）

要点是教学性的：scaffolding 是承重的。OpenHands
(arXiv:2407.16741) 明确下了 CodeAct 赌注；JSON tool calls
在提供者控制执行器的托管服务中占主导。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


# ---------- 迷你世界：一个微型内存 "仓库" ----------

INITIAL_REPO = {
    "app.py": "def add(a, b):\n    return a - b\n",
    "util.py": "def lower(s):\n    return s.upper()\n",
    "cli.py": "VERSION = 'v0.0'\n",
}

TESTS = [
    ("app.py", "add(2, 3) == 5"),
    ("util.py", "lower('AB') == 'ab'"),
    ("cli.py", "VERSION == 'v1.0'"),
]

# 存根 "模型" 在测试失败时应用的每路径替换。
# 集中化该表可避免在两种 scaffold 间重复 if/elif 链，
# 并在 TESTS 后续增长时避免 UnboundLocalError。
FIXES: dict[str, tuple[str, str]] = {
    "app.py": ("a - b", "a + b"),
    "util.py": ("s.upper()", "s.lower()"),
    "cli.py": ("v0.0", "v1.0"),
}


def run_tests(repo: dict[str, str]) -> list[bool]:
    """确定性存根：针对仓库字符串模拟测试套件。"""
    results = []
    for path, _expr in TESTS:
        src = repo.get(path, "")
        passed = False
        if path == "app.py":
            passed = "return a + b" in src
        elif path == "util.py":
            passed = "return s.lower()" in src
        elif path == "cli.py":
            passed = "VERSION = 'v1.0'" in src
        results.append(passed)
    return results


def _apply_fix(repo: dict[str, str], path: str) -> bool:
    """原地应用每路径修复。仅当修复被应用时返回 True。"""
    rule = FIXES.get(path)
    if rule is None:
        return False
    old, new = rule
    repo[path] = repo[path].replace(old, new)
    return True


# ---------- JSON tool-call scaffold：每轮一个动作 ----------

@dataclass
class JsonScaffold:
    repo: dict[str, str] = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0

    def step(self) -> str:
        """每次返回一个 JSON 动作，基于当前失败的测试。"""
        self.turns += 1
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results, strict=True):
            if ok:
                continue
            if _apply_fix(self.repo, path):
                return json.dumps({"tool": "edit", "path": path})
        return json.dumps({"tool": "done"})

    def blast_radius(self) -> int:
        return 1  # each action touches exactly one file

    def run(self, max_turns: int = 10) -> tuple[int, int]:
        for _ in range(max_turns):
            action = self.step()
            if json.loads(action).get("tool") == "done":
                break
        passed = sum(run_tests(self.repo))
        return passed, self.turns


# ---------- CodeAct scaffold：一个代码片段可能触及多个文件 ----------

@dataclass
class CodeActScaffold:
    repo: dict[str, str] = field(default_factory=lambda: dict(INITIAL_REPO))
    turns: int = 0
    # 追踪单个动作触及的观察到的最大文件数。
    # 这比 len(repo) 的静态上界更诚实，因为
    # it would not silently inflate if someone adds an untested helper.
    worst_touched: int = 0

    def step(self) -> str:
        """返回一个可能一次性编辑多个文件的 Python 代码片段。"""
        self.turns += 1
        # 单个 "snippet" 动作一次性重写每个失败的文件。
        snippet_lines = []
        results = run_tests(self.repo)
        for (path, _), ok in zip(TESTS, results, strict=True):
            if ok:
                continue
            if _apply_fix(self.repo, path):
                snippet_lines.append(f"fs.write('{path}', ...)")
        self.worst_touched = max(self.worst_touched, len(snippet_lines))
        if not snippet_lines:
            return "done()"
        return "; ".join(snippet_lines)

    def blast_radius(self) -> int:
        # 观察到的最坏情况：单个动作触及的文件数。
        return self.worst_touched

    def run(self, max_turns: int = 10) -> tuple[int, int]:
        for _ in range(max_turns):
            action = self.step()
            if action == "done()":
                break
        passed = sum(run_tests(self.repo))
        return passed, self.turns


# ---------- 驱动 ----------

def report(name: str, passed: int, turns: int, blast: int) -> None:
    total = len(TESTS)
    print(f"  {name:<18}  passed {passed}/{total}  turns {turns:>2}  "
          f"blast-radius {blast}")


def main() -> None:
    print("=" * 70)
    print("CODEACT vs JSON TOOL-CALL SCAFFOLDS (Phase 15, Lesson 9)")
    print("=" * 70)
    print()
    print("相同存根模型，三个 bug 的玩具仓库。仅 scaffold 对比。")
    print("-" * 70)

    js = JsonScaffold()
    passed, turns = js.run()
    report("JSON tool-call", passed, turns, js.blast_radius())

    ca = CodeActScaffold()
    passed, turns = ca.run()
    report("CodeAct (stub)", passed, turns, ca.blast_radius())

    print()
    print("=" * 70)
    print("HEADLINE: scaffold 不是布景。它就是产品本身。")
    print("-" * 70)
    print("  相同模型，两种 scaffold，不同轮数。")
    print("  CodeAct 将多次编辑压缩为一个动作。")
    print("  代价是爆炸半径：CodeAct 需要加固的 sandbox")
    print("  隔离（OpenHands 使用 Docker）。JSON tool-calls 通过构造获得安全，")
    print("  因为每个动作都被独立验证。")
    print("  两者没有严格优劣；权衡在于审计什么。")


if __name__ == "__main__":
    main()
