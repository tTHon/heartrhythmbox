import pathlib
import platform

# หลอกระบบว่า PosixPath คือ WindowsPath เพื่อให้โหลดโมเดลข้าม OS ได้
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

from fastai.vision.all import *

# 2. กำหนดชื่อไฟล์โมเดล (ตรวจสอบชื่อไฟล์ในเครื่องคุณให้ตรงกัน)
file_manuf = 'cied/classification_manuf.pkl'
file_model = 'cied/classification_model.pkl'

try:
    # 3. โหลดโมเดลทั้งสองตัว
    print("Loading models...")
    learn_manuf = load_learner(file_manuf)
    learn_model = load_learner(file_model)

    # 4. โหลดและเตรียมรูปภาพ
    img_path = 'cied/Dataset/VVI.jpg'
    img = PILImage.create(img_path)

    # 5. ทำนายผล
    # ทายยี่ห้อ
    manuf_name, _, manuf_probs = learn_manuf.predict(img)
    manuf_conf = manuf_probs.max() * 100

    # ทายรุ่น
    model_name, _, model_probs = learn_model.predict(img)
    model_conf = model_probs.max() * 100

    # 6. แสดงผลลัพธ์แบบสรุป
    print("-" * 30)
    print(f"IMAGE: {img_path}")
    print(f"MANUFACTURER : {manuf_name} ({manuf_conf:.2f}%)")
    print(f"MODEL GROUP  : {model_name} ({model_conf:.2f}%)")
    print("-" * 30)

except FileNotFoundError as e:
    print(f"Error: หาไฟล์โมเดลไม่พบ ตรวจสอบว่า {file_manuf} และ {file_model} อยู่ในโฟลเดอร์เดียวกันกับ code")
except Exception as e:
    print(f"An error occurred: {e}")