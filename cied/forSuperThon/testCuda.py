import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
# ทดสอบสร้าง Tensor เล็กๆ บน GPU
x = torch.rand(5, 3).cuda()
print("Test Tensor on GPU:", x.device)