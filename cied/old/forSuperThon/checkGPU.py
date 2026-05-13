import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ทดสอบส่งงานไปคำนวณจริง (จุดชี้ชะตา)
try:
    x = torch.rand(3, 3).to('cuda')
    y = x @ x
    print("🚀 Result: GPU Calculation SUCCESS!")
except Exception as e:
    print(f"❌ Result: GPU still NOT compatible. Error: {e}")