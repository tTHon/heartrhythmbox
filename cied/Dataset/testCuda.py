import torch

# เช็คว่า PyTorch ใช้ CUDA ได้ไหม
print(f"CUDA Available: {torch.cuda.is_available()}")

# เช็คเวอร์ชัน cuDNN ที่กำลังถูกเรียกใช้
if torch.cuda.is_available():
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    # เปิดใช้งาน Benchmark เพื่อให้ GTX 1650 เลือกอัลกอริทึมที่เร็วที่สุด
    torch.backends.cudnn.benchmark = True