"""Role specialization (角色特化): planner, executor, critic, verifier.

构建一个小型 Python 函数。Critic (LLM 模拟) 和 verifier (代码)
共同捕捉到单独任何一个都会遗漏的 bug。

运行两次：一次使用正确的 executor 输出，一次使用 off-spec (偏离规格) 输出。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Spec:
    task_name: str
    signature: str
    description: str
    tests: list[tuple[tuple, int]]


@dataclass
class Artifact:
    code: str


@dataclass
class CriticReport:
    approved: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class VerifierReport:
    passed: bool
    failures: list[str] = field(default_factory=list)


def planner(user_wish: str) -> Spec:
    """从高层愿望生成结构化 spec (规格说明)。"""
    return Spec(
        task_name="add_two",
        signature="add_two(a: int, b: int) -> int",
        description=user_wish,
        tests=[((1, 2), 3), ((10, 20), 30), ((-5, 5), 0)],
    )


def executor_correct(spec: Spec) -> Artifact:
    return Artifact(code="def add_two(a, b):\n    return a + b\n")


def executor_buggy(spec: Spec) -> Artifact:
    return Artifact(code="def add_two(a, b):\n    return a * b\n")


def critic(spec: Spec, art: Artifact) -> CriticReport:
    """LLM 风格审查。对常见问题进行模式匹配，但可能被
    看起来合理但语义错误的代码欺骗。"""
    notes: list[str] = []
    if "def" not in art.code:
        notes.append("missing def statement")
    if "return" not in art.code:
        notes.append("missing return")
    if spec.task_name not in art.code:
        notes.append(f"function name does not match spec '{spec.task_name}'")
    approved = not notes
    return CriticReport(approved=approved, notes=notes)


def verifier(spec: Spec, art: Artifact) -> VerifierReport:
    """在 sandbox namespace (沙箱命名空间) 中运行代码并执行测试。确定性的。"""
    ns: dict = {}
    try:
        exec(art.code, ns, ns)
    except Exception as e:
        return VerifierReport(passed=False, failures=[f"exec error: {e}"])
    fn = ns.get(spec.task_name)
    if not callable(fn):
        return VerifierReport(passed=False, failures=[f"no callable '{spec.task_name}' produced"])
    failures: list[str] = []
    for args, expected in spec.tests:
        try:
            got = fn(*args)
        except Exception as e:
            failures.append(f"call {args} raised {e}")
            continue
        if got != expected:
            failures.append(f"call {args}: expected {expected}, got {got}")
    return VerifierReport(passed=not failures, failures=failures)


def run_pipeline(user_wish: str, executor, label: str) -> None:
    print(f"\n=== {label} ===")
    spec = planner(user_wish)
    print(f"  [planner] spec: {spec.signature} with {len(spec.tests)} tests")
    art = executor(spec)
    print(f"  [executor] produced:\n    {art.code.replace(chr(10), chr(10)+'    ')}")
    crep = critic(spec, art)
    print(f"  [critic] approved={crep.approved}, notes={crep.notes}")
    vrep = verifier(spec, art)
    print(f"  [verifier] passed={vrep.passed}, failures={vrep.failures}")
    if crep.approved and vrep.passed:
        print("  RESULT: ship it.")
    elif not vrep.passed:
        print("  RESULT: verifier blocked ship (deterministic catch).")
    elif not crep.approved:
        print("  RESULT: critic blocked ship (subjective catch).")


def main() -> None:
    print("Role specialization pipeline — planner, executor, critic, verifier")
    print("-" * 70)

    run_pipeline(
        "A function that returns the sum of two integers.",
        executor_correct,
        "Correct executor output",
    )

    run_pipeline(
        "A function that returns the sum of two integers.",
        executor_buggy,
        "Buggy executor output (looks plausible; fails runtime)",
    )

    print("\n关键洞察: critic 通过了有 bug 的代码，因为它看起来没问题。")
    print("只有 verifier -- 确定性测试执行 -- 捕捉到了语义 bug。")
    print("全 LLM pipeline (没有 verifier) 会交付这个 bug。经典的 MAST 失效模式。")


if __name__ == "__main__":
    main()
