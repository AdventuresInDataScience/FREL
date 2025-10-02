"""
PyTorch GPU Detection Test
Tests if PyTorch can see and use the RTX 3080 GPU
"""

import torch

print("=" * 70)
print("PyTorch GPU Test")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Quick performance test
    print("\n" + "=" * 70)
    print("Performance Test: 5000x5000 Matrix Multiplication")
    print("=" * 70)
    
    # CPU test
    import time
    x_cpu = torch.randn(5000, 5000)
    start = time.time()
    y_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start
    print(f"\nCPU time: {cpu_time:.3f} seconds")
    
    # GPU test
    x_gpu = torch.randn(5000, 5000).cuda()
    torch.cuda.synchronize()  # Wait for GPU initialization
    start = time.time()
    y_gpu = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()  # Wait for computation to finish
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.3f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
    
    print("\n" + "=" * 70)
    print("✓ GPU IS WORKING! Your RTX 3080 is ready to use!")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("✗ NO GPU DETECTED")
    print("=" * 70)
    print("\nTroubleshooting:")
    print("1. Make sure PyTorch was installed with CUDA support")
    print("2. Check nvidia-smi shows your GPU")
    print("3. Restart your terminal/computer if just installed PyTorch")
