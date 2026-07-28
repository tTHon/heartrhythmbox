"""
predict_manuf.py
================
ใช้โมเดลที่ finetune แล้วทำนาย manufacturer จากภาพ CXR

วิธีใช้:
  python predict_manuf.py --model finetuned_manuf.pkl \
                          --orig  classification_manuf.pkl \
                          --img   patient_cxr.jpg

  # ทำนายทั้ง folder:
  python predict_manuf.py --model finetuned_manuf.pkl \
                          --orig  classification_manuf.pkl \
                          --folder test_images/
"""

import argparse
import pickle
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

INFER_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(finetuned_pkl: str, original_pkl: str):
    from finetune_manuf import load_finetuned
    model, orig_vocab, our_classes = load_finetuned(finetuned_pkl, original_pkl)
    model = model.to(DEVICE)
    # idx → class name  (เฉพาะ class ที่เรามี)
    idx_to_class = {v: k for k, v in orig_vocab.items() if k in our_classes}
    return model, idx_to_class, our_classes


def predict_image(model, img_path: str, idx_to_class: dict, top_k: int = 4):
    img    = Image.open(img_path).convert("RGB")
    tensor = INFER_TF(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(min(top_k, len(idx_to_class)))

    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        results.append((idx_to_class[idx], prob))
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict CIED manufacturer")
    parser.add_argument("--model",  required=True, help="finetuned_manuf.pkl")
    parser.add_argument("--orig",   required=True, help="classification_manuf.pkl (โมเดลเดิม)")
    parser.add_argument("--img",    default=None,  help="path to single image")
    parser.add_argument("--folder", default=None,  help="folder ที่มีภาพหลายภาพ")
    args = parser.parse_args()

    print(f"โหลดโมเดล...")
    model, idx_to_class, our_classes = load_model(args.model, args.orig)
    print(f"  our classes  : {our_classes}")
    print(f"  idx_to_class : {idx_to_class}\n")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    if args.img:
        paths = [Path(args.img)]
    elif args.folder:
        paths = [p for p in Path(args.folder).iterdir() if p.suffix.lower() in img_exts]
        paths.sort()
    else:
        print("ระบุ --img หรือ --folder ด้วยครับ")
        return

    print(f"{'ไฟล์':<35} {'Prediction':<20} {'Confidence':>10}")
    print("─" * 70)
    for p in paths:
        try:
            results = predict_image(model, str(p), idx_to_class)
            top_cls, top_prob = results[0]
            print(f"{p.name:<35} {top_cls:<20} {top_prob:>9.1%}")
            for cls, prob in results[1:]:
                print(f"  {'':33} {cls:<20} {prob:>9.1%}")
            print()
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")


if __name__ == "__main__":
    main()