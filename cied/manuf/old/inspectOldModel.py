import torch
import pathlib
import torch.nn as nn

# แก้ปัญหา NotImplementedError: cannot instantiate 'PosixPath'
# โดยการ Map PosixPath ให้กลายเป็น WindowsPath ชั่วคราว
pathlib.PosixPath = pathlib.WindowsPath
# 1. โหลดไฟล์ด้วย torch.load
# ใช้ weights_only=False เพราะเราต้องการดูโครงสร้าง Object ทั้งหมด
# map_location='cpu' เพื่อป้องกัน Error กรณีโมเดลเดิมเซฟมาจาก GPU
MODEL_PATH = "cied/classification_model.pkl"
data = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print(f"Type of loaded object: {type(data)}")

# ... (ส่วนบนคงเดิมจนถึงการโหลด data) ...

if isinstance(data, dict):
    # กรณีเป็น dictionary ทั่วไป
    for k in ['vocab', 'classes', 'idx_to_class']:
        if k in data:
            print(f"✅ Found {k}: {data[k]}")
else:
    print("\n--- FastAI Model Inspection ---")
    # 1. ตรวจสอบ dls.vocab (สำหรับ FastAI Learner)
    if hasattr(data, 'dls') and hasattr(data.dls, 'vocab'):
        print("✅ Found Vocab in dls.vocab:")
        for i, name in enumerate(data.dls.vocab):
            print(f"  Index {i}: {name}")
            
    # 2. ตรวจสอบ attribute อื่นๆ ที่อาจเก็บชื่อคลาส
    else:
        found_vocab = False
        for attr in dir(data):
            if 'vocab' in attr.lower() or 'classes' in attr.lower():
                val = getattr(data, attr)
                print(f"✅ Found potential vocab in '{attr}': {val}")
                found_vocab = True
        
        if not found_vocab:
            print("❌ ไม่พบตัวแปรที่เก็บรายชื่อคลาสในไฟล์นี้")

# ตรวจสอบขนาด Output ของ Layer สุดท้ายเพื่อยืนยัน
try:
    model_inner = data.model if hasattr(data, 'model') else data
    # เข้าถึง layer สุดท้ายตามโครงสร้าง [1][8]
    last_layer = model_inner[1][8] 
    print(f"\nModel Output Nodes: {last_layer.out_features}")
except:
    print("\nCould not verify last layer output features.")