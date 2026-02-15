#!/usr/bin/env python
"""CUDA 环境测试脚本

使用方法:
    conda activate cv
    python scripts/test_cuda.py
"""

import sys

def test_cuda():
    print("=" * 50)
    print("CUDA 环境测试")
    print("=" * 50)

    # 1. 检查 PyTorch
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyTorch not installed: {e}")
        return False

    # 2. 检查 CUDA 可用性
    print(f"\nCUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available")
        print("\n可能原因:")
        print("  1. NVIDIA 驱动未安装或版本不兼容")
        print("  2. PyTorch 安装的是 CPU 版本")
        print("  3. CUDA toolkit 版本与 PyTorch 不匹配")
        return False

    # 3. CUDA 详细信息
    print(f"\n[OK] CUDA version: {torch.version.cuda}")
    print(f"[OK] cuDNN version: {torch.backends.cudnn.version()}")
    print(f"[OK] Device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - Compute capability: {props.major}.{props.minor}")

    # 4. 简单计算测试
    print("\n" + "-" * 50)
    print("GPU 计算测试...")

    try:
        # 分配显存
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        # 矩阵乘法
        z = torch.matmul(x, y)
        torch.cuda.synchronize()

        print(f"[OK] 矩阵乘法测试通过")

        # 显存使用
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[OK] 显存使用: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")

        # 清理
        del x, y, z
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[FAIL] GPU 计算测试失败: {e}")
        return False

    # 5. AMP 测试
    print("\n" + "-" * 50)
    print("AMP (混合精度) 测试...")

    try:
        with torch.cuda.amp.autocast():
            x = torch.randn(100, 100, device='cuda')
            y = torch.matmul(x, x)
        print(f"[OK] AMP 测试通过")
    except Exception as e:
        print(f"[WARN] AMP 测试失败: {e}")

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)
