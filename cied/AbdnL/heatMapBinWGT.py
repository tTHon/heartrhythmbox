import pathlib
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# ==========================================================
# 1. SETTINGS & PARAMETERS
# ==========================================================
path_img_folder  = "C:/CIEDID_data/AbdnL/test_data"
path_mask_folder = "C:/CIEDID_data/AbdnL/test_mask"
path_weights     = "C:/CIEDID_data/AbdnL/models/best_abdn.pth"

filename = "a_x15.png"

def mask_path_for(img_path: pathlib.Path) -> pathlib.Path:
    candidates = [
        pathlib.Path(path_mask_folder) / f"{img_path.stem}_mask.png",
        pathlib.Path(path_mask_folder) / f"{img_path.stem}.png",
    ]
    for c in candidates:
        if c.exists(): return c
    raise FileNotFoundError(f"ไม่พบ mask สำหรับ {img_path.name}")

IMG_Size   = 640
threshold_high = 0.8    # เกณฑ์หลักในการตัดสินว่าพบ (Found)
threshold_low  = 0.5    # เกณฑ์ขั้นต่ำสำหรับพื้นที่เฝ้าระวัง (Attention Zone)
pixel_min      = 600    # ใช้เกณฑ์ความมั่นใจสูงในการตัดสินใจ True/False Positive

MASK_ALPHA_SOLID = 0.5  # ความทึบสำหรับคลาสทั่วไป และ High-confidence Abdn
MASK_ALPHA_LOW   = 0.2 # ความทึบจางพิเศษสำหรับพื้นที่ความมั่นใจต่ำ (0.5 - 0.8)

#COLOR_GENERATOR  = np.array([0, 1, 0])      # เขียว
#COLOR_LEAD       = np.array([0, 0.6, 1])    # ฟ้า

# กำหนด HEX Code ตายตัวสำหรับคลาสเพื่อความแม่นยำของสีทั้งในรูปและ Legend
HEX_GENERATOR    = "#00FF00"
HEX_LEAD         = "#0099FF"
HEX_ABDN_HIGH    = "#FF3333"  # สีแดงสดทึบ (High Confidence)
HEX_ABDN_LOW     = "#FF9900"  # สีส้มสว่างทึบ (Attention Zone)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# 2. LOAD MODEL
# ==========================================================
img_files = get_image_files(path_img_folder)
dls = SegmentationDataLoaders.from_label_func(
    pathlib.Path("."), bs=1, fnames=img_files[:1],
    label_func=lambda x: x, codes=["Background", "Generator", "Lead", "Abdn_Lead"],
    item_tfms=Resize(IMG_Size, method='pad', pad_mode='zeros')
)
learn = unet_learner(dls, resnet50, n_out=4)
learn.model.load_state_dict(torch.load(path_weights, map_location=device))
learn.model.to(device).eval()

timg_pipe = Pipeline([PILImage.create, Resize(IMG_Size, method='pad', pad_mode='zeros'), ToTensor(), IntToFloatTensor()])
mean = torch.tensor([0.5150052309036255] * 3, device=device).view(3, 1, 1)
std  = torch.tensor([0.23788487911224365] * 3, device=device).view(3, 1, 1)

# ปรับปรุงฟังก์ชันวาด Mask ให้รองรับสองระดับความมั่นใจของ Abandoned Lead
def draw_masks(ax, generator_mask, lead_mask, abdn_high, abdn_low):
    masks_definition = [
        (generator_mask, HEX_GENERATOR ),
        (lead_mask, HEX_LEAD),
        (abdn_low, HEX_ABDN_LOW),       # วาดพื้นที่ความมั่นใจต่ำก่อน (อยู่ชั้นล่าง)
        (abdn_high, HEX_ABDN_HIGH)     # วาดพื้นที่ความมั่นใจสูงทับด้านบน
    ]
    for mask, hex_color in masks_definition:
        if mask.sum() == 0: continue
        # สร้าง Custom Colormap ที่มีสีเดี่ยวทึบ
        cm = ListedColormap(['#00000000', hex_color])
        ax.imshow(mask, cmap=cm, alpha=0.55)

# ==========================================================
# 3. FIND IMAGE
# ==========================================================
matches = [p for p in img_files if p.name == filename]
if not matches: raise FileNotFoundError(f"ไม่พบไฟล์ {filename}")
img_path = matches[0]

# ==========================================================
# 4. INFERENCE & TWO-TIER THRESHOLDING
# ==========================================================
timg = timg_pipe(img_path).to(device)
timg_norm = (timg - mean) / std

with torch.no_grad():
    output = learn.model(timg_norm.unsqueeze(0))
    probs = F.softmax(output, dim=1)[0].cpu().numpy()

abdn_prob = probs[3]
most_likely_class = probs.argmax(axis=0)

# แยกพิกเซลออกเป็น 2 ระดับความมั่นใจ
abdn_mask_high = abdn_prob > threshold_high
abdn_mask_low  = (abdn_prob >= threshold_low) & (abdn_prob <= threshold_high)

# คำนวณหาจำนวนพิกเซลรวมในแต่ละเซกเมนต์เพื่อนำไปโชว์บนหัวรูป
pixel_count_high = int(abdn_mask_high.sum())
pixel_count_low  = int(abdn_mask_low.sum())

# ลบพื้นที่ทับซ้อนออกจากคลาสปกติ
generator_mask_pred = (most_likely_class == 1) & (~abdn_mask_high) & (~abdn_mask_low)
lead_mask_pred      = (most_likely_class == 2) & (~abdn_mask_high) & (~abdn_mask_low)

pred_detected = pixel_count_high > pixel_min
raw_img = timg.permute(1, 2, 0).cpu().numpy()

# ==========================================================
# 5. LOAD GROUND TRUTH
# ==========================================================
gt_path = mask_path_for(img_path)
raw_pil_img  = PILImage.create(img_path)
raw_pil_mask = PILMask.create(gt_path)

if raw_pil_img.size != raw_pil_mask.size:
    raw_pil_mask = raw_pil_mask.resize(raw_pil_img.size, resample=Image.NEAREST)

mask_resize = Resize(IMG_Size, method='pad', pad_mode='zeros')
gt_mask = np.array(mask_resize(raw_pil_mask))
if gt_mask.ndim == 3: gt_mask = gt_mask[..., 0]

gt_abdn_mask      = (gt_mask == 3)
gt_lead_mask      = (gt_mask == 2)
gt_generator_mask = (gt_mask == 1)
gt_has_abandoned  = gt_abdn_mask.sum() > 0

# ==========================================================
# 6. PLOT — MANUSCRIPT BLACK STYLE (WITH TWO-TIER METRICS)
# ==========================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter'] + plt.rcParams['font.sans-serif']
plt.rcParams['text.color'] = '#FFFFFF'
plt.rcParams['axes.labelcolor'] = '#FFFFFF'

if pred_detected and gt_has_abandoned:
    agreement, ag_color = "TRUE POSITIVE", "#2ecc71"
elif pred_detected and not gt_has_abandoned:
    agreement, ag_color = "FALSE POSITIVE", "#f39c12"
elif not pred_detected and gt_has_abandoned:
    agreement, ag_color = "FALSE NEGATIVE", "#e74c3c"
else:
    agreement, ag_color = "TRUE NEGATIVE", "#2ecc71"

gt_text = "PRESENT" if gt_has_abandoned else "ABSENT"
gt_color = '#e74c3c' if gt_has_abandoned else '#95a5a6'
pred_text = "FOUND" if pred_detected else "NOT FOUND"
pred_color = '#e74c3c' if pred_detected else '#95a5a6'

fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 8), facecolor='#000000')

# --- Left: GROUND TRUTH ---
ax_gt.set_facecolor('#000000')
ax_gt.imshow(raw_img)
ax_gt.set_title(f"GROUND TRUTH\nAbdn Lead: {gt_text} ({int(gt_abdn_mask.sum()):,} px)",
                 fontsize=12, color=gt_color, fontweight='bold', pad=12)
ax_gt.axis('off')
ax_gt.text(0.01, 0.99, "(A)", transform=ax_gt.transAxes, color='#FFFFFF', fontsize=16, fontweight='bold', va='top')

# --- Right: PREDICTION ---
ax_pred.set_facecolor('#000000')
ax_pred.imshow(raw_img)
# ส่งค่าทั้งขอบเขต high และ low เข้าไปวาดพร้อมกัน
draw_masks(ax_pred, generator_mask_pred, lead_mask_pred, abdn_mask_high, abdn_mask_low)
ax_pred.set_title(f"PREDICTION\nAbdn Lead: {pred_text} {pixel_count_high:,} px",
                   fontsize=12, color=pred_color, fontweight='bold', pad=12)
ax_pred.axis('off')
ax_pred.text(0.01, 0.99, "(B)", transform=ax_pred.transAxes, color='#FFFFFF', fontsize=16, fontweight='bold', va='top')

# จัดวางสถานะหลักและเส้นแบ่งกลางรูปภาพตามสไตล์ Manuscript
fig.text(0.5, 0.09, f"{agreement} for abandoned lead detection", ha='center', va='center', fontsize=15, color=ag_color, fontweight='bold',
         bbox=dict(facecolor='#111111', edgecolor=ag_color, boxstyle='round,pad=0.3', linewidth=0.5))
fig.add_artist(plt.Line2D([0.5, 0.5], [0.18, 0.84], color='#333333', linewidth=1.2))

# ==========================================================
# 7. LEGEND & SAVE
# ==========================================================
# เพิ่มคำอธิบายลักษณะของระดับสีทั้งสองใน Legend บรรทัดล่าง
custom_lines = [
    Line2D([0], [0], color=HEX_GENERATOR, lw=4),
    Line2D([0], [0], color=HEX_LEAD, lw=4),
    Line2D([0], [0], color=HEX_ABDN_HIGH, lw=4),
    Line2D([0], [0], color=HEX_ABDN_LOW, lw=4)
]

leg = fig.legend(
    custom_lines,
    [
        'Generator', 
        'Normal Lead', 
        f'Abandoned Lead (Probability > {threshold_high})',
        f'Abandoned Lead (Probability 0.5-0.8))'
    ],
    loc='lower center', 
    ncol=2,  # ปรับเป็น 2x2 แนวนอนเพื่อให้อ่านข้อความที่ยาวขึ้นได้ง่ายและเป็นระเบียบ
    fontsize=9.5, 
    frameon=True,
    facecolor='#111111',  
    edgecolor='#333333',  
    labelcolor='#FFFFFF'  
)
leg.get_frame().set_linewidth(0.8)

plt.tight_layout(rect=[0, 0.12, 1, 0.90])
plt.savefig("cied/AbdnL/heatmap_allClasses.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')