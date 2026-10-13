"""
PyTorch CUDA 环境冒烟测试 (Smoke Test)
用于快速验证 CUDA 环境是否正常工作
"""

import sys

import torch
from torch import nn


def print_separator(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_cuda_availability():
    """测试 1: CUDA 是否可用及基本信息"""
    print_separator("Test 1: CUDA Availability")

    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    assert available, "❌ CUDA 不可用，请检查驱动和 PyTorch 安装"

    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA version:     {torch.version.cuda}")
    print(f"cuDNN version:    {torch.backends.cudnn.version()}")
    print(f"Device count:     {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU[{i}]: {props.name}")
        print(f"          显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"          计算能力: {props.major}.{props.minor}")
        print(f"          多处理器数量: {props.multi_processor_count}")

    print("✅ Test 1 PASSED")


def test_basic_tensor_ops():
    """测试 2: 基础张量运算"""
    print_separator("Test 2: Basic Tensor Operations")

    device = torch.device("cuda:0")

    # 创建张量并移动到 GPU
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # 矩阵乘法
    c = a @ b
    print(f"Matrix multiply result shape: {c.shape}")
    print(f"Result device: {c.device}")

    # 验证设备
    assert c.device.type == "cuda", "❌ 张量未在 GPU 上"
    assert torch.allclose(c.cpu(), (a.cpu() @ b.cpu()), atol=1e-4), "❌ 结果不一致"

    # 常见运算
    d = torch.relu(c) + torch.sigmoid(c) * 2.0
    e = d.sum().item()
    print(f"Scalar result: {e:.4f}")

    # CPU <-> GPU 传输
    cpu_tensor = torch.ones(3, 3)
    gpu_tensor = cpu_tensor.to(device)
    back_to_cpu = gpu_tensor.cpu()
    assert torch.equal(cpu_tensor, back_to_cpu), "❌ 数据往返传输不一致"
    print("CPU <-> GPU transfer: OK")

    print("✅ Test 2 PASSED")


def test_neural_network():
    """测试 3: 简单神经网络前向 + 反向传播"""
    print_separator("Test 3: Neural Network Forward + Backward")

    device = torch.device("cuda:0")

    # 定义简单模型
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    # 假数据
    x = torch.randn(32, 128, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 前向
    logits = model(x)
    loss = criterion(logits, y)
    print(f"Initial loss: {loss.item():.4f}")

    # 反向
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 检查梯度是否有效
    for name, param in model.named_parameters():
        assert param.grad is not None, f"❌ {name} 梯度为 None"
        assert not torch.isnan(param.grad).any(), f"❌ {name} 梯度包含 NaN"
    print("Gradient check: OK")

    print("✅ Test 3 PASSED")


def test_determinism_and_sync():
    """测试 4: 同步与确定性"""
    print_separator("Test 4: Synchronization & Determinism")

    device = torch.device("cuda:0")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x = torch.randn(500, 500, device=device)
    y1 = x @ x.T

    torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
    print("torch.cuda.synchronize(): OK")

    # 再次计算相同数据（相同输入应得到相同结果）
    y2 = x @ x.T
    assert torch.allclose(y1, y2, atol=1e-5), "❌ 相同输入结果不一致"

    print("✅ Test 4 PASSED")


def test_amp():
    """测试 5: 混合精度 (AMP)"""
    print_separator("Test 5: Automatic Mixed Precision (AMP)")

    if not torch.cuda.is_bf16_supported() and not torch.cuda.is_available():
        print("⚠️  跳过: 不支持 AMP")
        return

    device = torch.device("cuda:0")

    model = nn.Linear(256, 256).to(device)
    x = torch.randn(64, 256, device=device)

    scaler = torch.amp.GradScaler("cuda")

    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = out.sum()

    scaler.scale(loss).backward()
    scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
    scaler.update()

    print(f"AMP loss: {loss.item():.4f}")
    print("✅ Test 5 PASSED")


def test_memory_info():
    """测试 6: 显存信息"""
    print_separator("Test 6: Memory Info")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2

    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved:  {reserved:.2f} MB")

    # 分配一个大张量，再查看
    big = torch.zeros(1000, 1000, 1000, device=device)  # ~4GB fp32
    allocated_after = torch.cuda.memory_allocated(device) / 1024**2
    print(f"After allocating 4GB tensor: {allocated_after:.2f} MB")

    del big
    torch.cuda.empty_cache()

    print("✅ Test 6 PASSED")


def main():
    print("\n" + "🔥" * 30)
    print("  PyTorch CUDA Smoke Test")
    print("🔥" * 30)

    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("Basic Tensor Ops", test_basic_tensor_ops),
        ("Neural Network", test_neural_network),
        ("Sync & Determinism", test_determinism_and_sync),
        ("AMP", test_amp),
        ("Memory Info", test_memory_info),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"❌ {name} FAILED: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"❌ {name} ERROR: {type(e).__name__}: {e}")
            failed += 1

    print_separator("Summary")
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    if failed == 0:
        print("🎉 All tests passed! CUDA environment is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()