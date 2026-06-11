# Lesson: GPU Setup & Cloud (phase 00 / lesson 03)
# 课程：GPU 配置与云计算（阶段 00 / 课程 03）
# Check GPU availability, run CPU vs GPU benchmark, estimate max model size.
# 检查 GPU 可用性，运行 CPU 与 GPU 基准测试，估算最大模型大小。

import time
import sys


def check_gpu():
    """Check GPU availability and run a CPU vs GPU matrix multiply benchmark.
    检查 GPU 可用性并运行 CPU 与 GPU 矩阵乘法基准测试。"""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")  # PyTorch 未安装，请运行 pip install torch
        return

    print("=== GPU Check ===\n")  # GPU 检查
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo GPU detected. That's fine for most lessons.")  # 未检测到 GPU，大多数课程不受影响
        print("For GPU-heavy lessons, use Google Colab (free).")  # GPU 密集型课程可使用免费的 Google Colab
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    print("\n=== CPU vs GPU Benchmark ===\n")  # CPU 与 GPU 基准测试
    size = 4000

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")
    torch.cuda.synchronize()

    start = time.time()
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")

    vram_gb = props.total_memory / 1e9
    params_fp16 = vram_gb * 1e9 / 2
    params_billions = params_fp16 / 1e9
    print(f"\nEstimated max model size (fp16): ~{params_billions:.0f}B parameters")  # fp16 下估算的最大模型大小


if __name__ == "__main__":
    check_gpu()
