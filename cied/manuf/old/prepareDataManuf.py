"""
prepare_dataset.py
==================
จัดโครงสร้างภาพก่อน finetune

วิธีใช้:
  1. วางภาพทั้งหมดใน raw_images/ โดยให้ชื่อไฟล์มี manufacturer อยู่ด้วย
     เช่น  Abbott_001.jpg, Medtronic_042.png  ฯลฯ
     
     หรือ ถ้าภาพอยู่ใน subfolder แยกตาม manufacturer อยู่แล้ว ให้ใช้ --from_folders

  2. รัน:
     python prepare_dataset.py --img_dir raw_images/ --out_dir dataset/
     
     หรือ ถ้าจัด folder แล้ว:
     python prepare_dataset.py --from_folders raw_images/ --out_dir dataset/

โครงสร้าง output:
  dataset/
    train/
      Abbott/
      Medtronic/
      BostonScientific/
      Biotronik/
    val/
      Abbott/
      ...
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

CLASSES = ["Abbott", "Medtronic", "BostonScientific", "Biotronik"]

# คำที่ใช้ match ชื่อไฟล์ → class (case-insensitive)
CLASS_KEYWORDS = {
    "Abbott":           ["abbott", "st. jude", "stjude"],
    "Medtronic":        ["medtronic", "mdt"],
    "BostonScientific": ["boston", "bsc", "bostonscientific"],
    "Biotronik":        ["biotronik", "biot"],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def guess_class_from_filename(filename: str) -> str | None:
    name_lower = filename.lower()
    for cls, keywords in CLASS_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cls
    return None


def prepare_from_flat(img_dir: Path, out_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    """ภาพอยู่ใน folder เดียว ชื่อไฟล์บอก class"""
    random.seed(seed)
    buckets = defaultdict(list)

    for p in img_dir.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            cls = guess_class_from_filename(p.name)
            if cls:
                buckets[cls].append(p)
            else:
                print(f"  [SKIP] ไม่รู้ class: {p.name}")

    _split_and_copy(buckets, out_dir, val_ratio)


def prepare_from_folders(src_dir: Path, out_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    """ภาพอยู่ใน subfolder ชื่อ folder = class (หรือใกล้เคียง)"""
    random.seed(seed)
    buckets = defaultdict(list)

    for folder in src_dir.iterdir():
        if not folder.is_dir():
            continue
        cls = guess_class_from_filename(folder.name) or folder.name
        if cls not in CLASSES:
            print(f"  [WARN] folder '{folder.name}' → ไม่ตรง class ที่กำหนด (ใช้ชื่อ folder เลย)")
        for p in folder.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                buckets[cls].append(p)

    _split_and_copy(buckets, out_dir, val_ratio)


def _split_and_copy(buckets: dict, out_dir: Path, val_ratio: float):
    print("\n=== สรุปก่อน split ===")
    for cls, files in sorted(buckets.items()):
        print(f"  {cls}: {len(files)} ภาพ")

    for split in ("train", "val"):
        for cls in CLASSES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print("\n=== copy ไฟล์ ===")
    for cls, files in buckets.items():
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for split, split_files in [("train", train_files), ("val", val_files)]:
            dst_dir = out_dir / split / cls
            for src in split_files:
                shutil.copy2(src, dst_dir / src.name)
        print(f"  {cls}: train={len(train_files)}, val={len(val_files)}")

    print(f"\n✅ dataset พร้อมแล้วที่ {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare manufacturer dataset")
    parser.add_argument("--img_dir", type=Path, default=None,
                        help="flat folder ที่มีภาพทั้งหมด (ชื่อไฟล์ต้องมี manufacturer)")
    parser.add_argument("--from_folders", type=Path, default=None,
                        help="source dir ที่มี subfolder แยกตาม manufacturer")
    parser.add_argument("--out_dir", type=Path, default=Path("dataset"),
                        help="output directory (default: dataset/)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="สัดส่วน validation set (default: 0.2)")
    args = parser.parse_args()

    if args.from_folders:
        prepare_from_folders(args.from_folders, args.out_dir, args.val_ratio)
    elif args.img_dir:
        prepare_from_flat(args.img_dir, args.out_dir, args.val_ratio)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()