# Lesson: Dev Environment (phase 00 / lesson 01)
# 课程：开发环境（阶段 00 / 课程 01）
# Verify that the four-layer toolchain (system, package managers, runtimes, libs)
# is reachable. Reports PASS/FAIL plus version details. Stdlib only.
# 验证四层工具链（系统、包管理器、运行时、库）是否可用。输出 PASS/FAIL 及版本详情。仅使用标准库。

import sys
import shutil
import subprocess

# Core checks: Python, libraries, and toolchain presence
# 核心检查项：Python、库及工具链是否存在
CHECKS = [
    ("Python 3.10+", lambda: sys.version_info >= (3, 10), f"Python {sys.version}"),
    ("NumPy", lambda: __import__("numpy"), None),
    ("Matplotlib", lambda: __import__("matplotlib"), None),
    ("Jupyter", lambda: __import__("jupyter"), None),
    ("Git", lambda: shutil.which("git") is not None, None),
    ("Node.js", lambda: shutil.which("node") is not None, None),
    ("Rust (cargo)", lambda: shutil.which("cargo") is not None, None),
]

# GPU checks: PyTorch and CUDA availability
# GPU 检查项：PyTorch 和 CUDA 是否可用
GPU_CHECKS = [
    ("PyTorch", lambda: __import__("torch"), None),
    (
        "CUDA",
        lambda: __import__("torch").cuda.is_available(),
        lambda: __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "Not available",
    ),
]


def run_check(name, check_fn, detail_fn=None):
    """Run a single check and print PASS/FAIL with optional detail.
    运行单条检查并打印 PASS/FAIL 及可选详情。"""
    try:
        result = check_fn()
        if result is False:
            raise Exception("Check returned False")
        detail = ""
        if detail_fn:
            if callable(detail_fn):
                detail = f" ({detail_fn()})"
            else:
                detail = f" ({detail_fn})"
        print(f"  [PASS] {name}{detail}")
        return True
    except Exception:
        print(f"  [FAIL] {name}")
        return False


def main():
    print("\n=== AI Engineering from Scratch — Environment Check ===\n")

    print("Core:")
    passed = sum(run_check(name, fn, detail) for name, fn, detail in CHECKS)
    total = len(CHECKS)

    print("\nGPU (optional):")  # GPU 为可选项
    gpu_passed = sum(run_check(name, fn, detail) for name, fn, detail in GPU_CHECKS)
    gpu_total = len(GPU_CHECKS)

    print(f"\nResult: {passed}/{total} core checks passed", end="")
    if gpu_passed > 0:
        print(f", {gpu_passed}/{gpu_total} GPU checks passed")
    else:
        print(" (no GPU — that's fine, most lessons work on CPU)")

    if passed == total:
        print("\nYou're ready. Start with Phase 1.\n")  # 环境就绪，从 Phase 1 开始
    else:
        print("\nFix the failed checks above, then run this script again.\n")  # 修复上方失败的检查项，然后重新运行

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
