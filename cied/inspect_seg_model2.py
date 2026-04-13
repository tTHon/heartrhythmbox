"""
รัน script นี้บนเครื่องเพื่อดูว่า key จริงๆ ใน segmentation.pkl คืออะไร
แล้วนำ prefix ที่ได้ไปแก้ ENCODER_PREFIXES ใน finetuneYN1.py

Usage:
    python inspect_pkl_keys.py
    python inspect_pkl_keys.py --model_path C:/path/to/your.pkl
"""

import pathlib
import platform
import argparse

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import load_learner

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="C:/CIEDID_data/pkl/segmentation.pkl")
args = parser.parse_args()

print(f"Loading: {args.model_path}")
learn = load_learner(args.model_path, cpu=True)
state = learn.model.state_dict()
keys  = list(state.keys())

print(f"\nTotal keys: {len(keys)}")
print("\n--- First 40 keys (shows prefix pattern) ---")
for k in keys[:40]:
    print(f"  {k:70s}  shape={tuple(state[k].shape)}")

print("\n--- Last 10 keys (shows head) ---")
for k in keys[-10:]:
    print(f"  {k:70s}  shape={tuple(state[k].shape)}")

# ดู unique top-level prefixes (ส่วนก่อน "." แรก)
prefixes = sorted(set(k.split(".")[0] for k in keys))
print(f"\n--- Unique top-level prefixes ---")
print(f"  {prefixes}")
print("\nนำ prefix ของ encoder ไปใส่ใน ENCODER_PREFIXES ใน finetuneYN1.py")
print("ปกติ ResNet50 encoder จะเป็น prefixes แรกๆ เช่น '0', '1', '2', '3', '4'")
print("ให้สังเกตว่า prefix ไหนที่ key มี 'layer' หรือ 'conv' หรือ 'bn' อยู่ใน path")