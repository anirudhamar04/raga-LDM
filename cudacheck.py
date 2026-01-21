import torch

print("=" * 50)
print("CUDA Availability Check")
print("=" * 50)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test a simple operation
    print("\n" + "=" * 50)
    print("Testing GPU with a simple tensor operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU test successful!")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("\nCUDA is not available.")
    print("Possible reasons:")
    print("  - PyTorch was installed without CUDA support")
    print("  - No NVIDIA GPU detected")
    print("  - CUDA drivers not installed")
    print("  - GPU not compatible with installed CUDA version")

print("=" * 50)
print(f"PyTorch Version: {torch.__version__}")
print("=" * 50)