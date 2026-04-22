# finetuneYN_custom.py  —  abandoned lead segmentation finetuning
#
# KEY DESIGN: PIL-level patch crop (RandomSplitCrop)
# ────────────────────────────────────────────────────
# FastAI's DataBlock pipeline:
#   1. load image as PIL  →  item_tfms  →  ToTensor  →  batch_tfms  →  GPU
#
# Our crop runs at step 1 (PIL level), BEFORE ToTensor, so:
#   • every image is already patch_size × patch_size when collation happens
#     → no "tensors not same size" error
#   • lead thickness is preserved (resize to std_size, then crop — no 1024→512 squeeze)
#
# RandomSplitCrop reads split_idx from the DataLoader to decide
# train (biased) vs val (center) crop automatically.

import pathlib, platform, argparse, random
import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImageLib
from fastai.vision.all import *
from sklearn.model_selection import train_test_split

# PyTorch 2.6+ fix
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False); return _orig_load(*a, **kw)
torch.load = _patched_load

warnings.filterwarnings("ignore")
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


# ══════════════════════════════════════════════════════
# 1. GLOBALS
# ══════════════════════════════════════════════════════
CLASS_NAMES  = ["background", "generator", "lead", "abandoned_lead"]
CLASS_COLORS = [(0.15,0.15,0.15),(0.20,0.60,1.00),(0.20,0.85,0.45),(1.00,0.25,0.25)]
N_CLASSES    = len(CLASS_NAMES)

def get_x(r): return r["image"]
def get_y(r): return r["mask"]

def dice_generator(inp, targ, eps=1e-6):
    pred  = inp.argmax(dim=1)
    p, t  = (pred==1).float(), (targ==1).float()
    return (2.*( p*t).sum()+eps) / (p.sum()+t.sum()+eps)

def abdn_lead_sensitivity(inp, targ):
    pred     = inp.argmax(dim=1)
    pred_yes = (pred==3).sum(dim=(1,2)) > 1
    targ_yes = (targ==3).sum(dim=(1,2)) > 0
    actual_pos = targ_yes.sum()
    if actual_pos == 0: return torch.tensor(float('nan'))
    return ((pred_yes & targ_yes).sum().float()) / actual_pos.float()


# ══════════════════════════════════════════════════════
# 2. PIL-LEVEL PATCH CROP  (the core fix)
# ══════════════════════════════════════════════════════
class RandomSplitCrop(ItemTransform):
    """
    PIL-level crop that runs INSIDE item_tfms, before ToTensor.

    For every (image_path, mask_path) pair FastAI loads both as PIL objects
    and passes them as a tuple to this transform.

    TRAIN (split_idx == 0):
        1. Resize image+mask to std_size × std_size (NEAREST for mask).
        2. With probability lead_sample_ratio, pick a crop center that
           lands on a lead or abandoned_lead pixel.
        3. Otherwise pick a uniformly random center.

    VAL (split_idx == 1):
        1. Same resize.
        2. Always center crop.

    Result: every item is exactly patch_size × patch_size — no collation error.
    """

    def __init__(self, patch_size=512, std_size=1024,
                 lead_sample_ratio=0.7, lead_classes=(2,3)):
        self.patch_size        = patch_size
        self.std_size          = std_size
        self.lead_sample_ratio = lead_sample_ratio
        self.lead_classes      = lead_classes

    def _lead_coords(self, mask_np):
        half   = self.patch_size // 2
        H, W   = mask_np.shape
        ys, xs = np.where(np.isin(mask_np, self.lead_classes))
        valid  = (ys>=half)&(ys<H-half)&(xs>=half)&(xs<W-half)
        return list(zip(ys[valid].tolist(), xs[valid].tolist()))

    def _box(self, H, W, mask_np, is_train):
        half = self.patch_size // 2
        cy = cx = None
        if is_train and random.random() < self.lead_sample_ratio:
            coords = self._lead_coords(mask_np)
            if coords: cy, cx = random.choice(coords)
        if cy is None:
            if is_train:
                cy = random.randint(half, max(half, H-half-1))
                cx = random.randint(half, max(half, W-half-1))
            else:
                cy, cx = H//2, W//2
        y0 = max(0, cy-half); y1 = y0+self.patch_size
        x0 = max(0, cx-half); x1 = x0+self.patch_size
        if y1>H: y0,y1 = H-self.patch_size, H
        if x1>W: x0,x1 = W-self.patch_size, W
        return x0, y0, x1, y1   # PIL crop expects (left,top,right,bottom)

    def encodes(self, xy):
        img_pil, mask_pil = xy   # PIL Image, PIL Image (mask)
        is_train = self.split_idx == 0

        # Resize to common size
        img_r  = img_pil.resize( (self.std_size, self.std_size), PILImageLib.BILINEAR)
        mask_r = mask_pil.resize((self.std_size, self.std_size), PILImageLib.NEAREST)

        mask_np = np.array(mask_r)
        H, W    = mask_np.shape
        box     = self._box(H, W, mask_np, is_train)

        return img_r.crop(box), mask_r.crop(box)


# ══════════════════════════════════════════════════════
# 3. AUTO CLASS WEIGHTS
# ══════════════════════════════════════════════════════
def compute_class_weights(df, n_classes=N_CLASSES, max_w=50.0):
    print("\n⚖️  Computing class weights …")
    px = np.zeros(n_classes, dtype=np.int64)
    for mp in df["mask"].unique():
        vals, cnts = np.unique(np.array(PILImage.create(mp)), return_counts=True)
        for v,c in zip(vals,cnts):
            if v < n_classes: px[v] += int(c)
    total   = px.sum()
    w       = total / (n_classes * np.maximum(px,1).astype(float))
    w       = w / w[0]
    w       = np.minimum(w, max_w)
    for i,(name,wi,pxi) in enumerate(zip(CLASS_NAMES,w,px)):
        print(f"  {name:20s}: {pxi:>12,} px ({100*pxi/total:5.2f}%)  → weight {wi:.2f}")
    return torch.tensor(w, dtype=torch.float32)


# ══════════════════════════════════════════════════════
# 4. DATASET SUMMARY
# ══════════════════════════════════════════════════════
def summarize_dataset(df):
    print("\n========== DATASET SUMMARY ==========")
    print(f"Total:{len(df)}  Train:{len(df[~df['is_valid']])}  Val:{len(df[df['is_valid']])}")
    cc = {i:0 for i in range(N_CLASSES)}
    for _,row in df.iterrows():
        for c in np.unique(np.array(PILImage.create(row["mask"]))):
            if int(c)<N_CLASSES: cc[int(c)]+=1
    for i,n in enumerate(CLASS_NAMES):
        print(f"  Class {i} ({n}): {cc[i]} images")
    print("=====================================\n")


# ══════════════════════════════════════════════════════
# 5. BATCH INSPECTION
# ══════════════════════════════════════════════════════
def _mask_rgb(arr):
    rgb = np.zeros((*arr.shape,3),float)
    for i,c in enumerate(CLASS_COLORS):
        for ch,v in enumerate(c): rgb[:,:,ch][arr==i]=v
    return rgb

def show_batch_inspection(dls, n=4, save_path=None, stats_mean=None, stats_std=None):
    if stats_mean is None: stats_mean=[0.485,0.456,0.406]
    if stats_std  is None: stats_std =[0.229,0.224,0.225]
    rows,labels=[],[]
    for sn,dl in [("TRAIN",dls.train),("VALID",dls.valid)]:
        xb,yb=next(iter(dl)); xb=xb.cpu(); yb=yb.cpu()
        for i in range(min(n,xb.shape[0])):
            img_t=xb[i]; msk=yb[i].squeeze().numpy()
            m=torch.tensor(stats_mean).view(3,1,1); s=torch.tensor(stats_std).view(3,1,1)
            img_np=(img_t*s+m).clamp(0,1).permute(1,2,0).numpy()
            ov=(0.55*img_np+0.45*_mask_rgb(msk)).clip(0,1)
            px=[(msk==c).sum() for c in range(N_CLASSES)]
            rows.append((img_np,ov,msk,px,msk.size)); labels.append(f"{sn}#{i+1}")
    ns=len(rows)
    fig,axes=plt.subplots(ns,3,figsize=(13,3.6*ns),gridspec_kw={"width_ratios":[1,1,0.9]})
    if ns==1: axes=[axes]
    for ri,((img_np,ov,msk,px,tot),lab) in enumerate(zip(rows,labels)):
        ai,ao,ab=axes[ri]
        ai.imshow(img_np); ai.set_title(f"{lab}\nraw image",fontsize=9,pad=4); ai.axis("off")
        ao.imshow(ov);     ao.set_title("mask overlay",fontsize=9,pad=4);      ao.axis("off")
        pcts=[100*c/tot for c in px]; yp=np.arange(N_CLASSES)
        bars=ab.barh(yp,pcts,color=CLASS_COLORS,height=0.55)
        ab.set_yticks(yp); ab.set_yticklabels(CLASS_NAMES,fontsize=8)
        ab.set_xlabel("% pixels",fontsize=8); ab.set_title("distribution",fontsize=9,pad=4)
        ab.invert_yaxis()
        for bar,pct in zip(bars,pcts):
            if pct>1: ab.text(pct+0.3,bar.get_y()+bar.get_height()/2,f"{pct:.1f}%",va="center",fontsize=7.5)
        ab.spines[["top","right"]].set_visible(False)
    patches=[mpatches.Patch(color=c,label=n) for c,n in zip(CLASS_COLORS,CLASS_NAMES)]
    fig.legend(handles=patches,loc="lower center",ncol=4,fontsize=9,frameon=False,bbox_to_anchor=(0.5,-0.01))
    fig.suptitle("Batch inspection — RandomSplitCrop (PIL-level)",fontsize=11,y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path,dpi=120,bbox_inches="tight")
        print(f"📊 Saved → {save_path}")
    else: plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════
# 6. DATAFRAME BUILDER
# ══════════════════════════════════════════════════════
def build_dataframe(args):
    rows=[]
    for img in args.new_imgs.iterdir():
        if img.suffix.lower() not in {".jpg",".png",".jpeg",".bmp"}: continue
        mask=args.new_masks/f"{img.stem}_mask.png"
        if not mask.exists(): mask=args.new_masks/f"{img.stem}.png"
        if not mask.exists(): continue
        arr=np.array(PILImage.create(mask))
        rows.append({"image":str(img.resolve()),"mask":str(mask.resolve()),
                     "has_abandoned":3 in np.unique(arr)})
    df=pd.DataFrame(rows)
    n_pos=df["has_abandoned"].sum()
    strat=df["has_abandoned"] if n_pos>=5 else None
    if strat is None: print("⚠️  Too few abandoned-lead images — random split.")
    ti,vi=train_test_split(df.index,test_size=args.valid_split,stratify=strat,random_state=42)
    df["is_valid"]=False; df.loc[vi,"is_valid"]=True
    # BUG FIX 1: oversample train only
    tr=df[~df["is_valid"]].copy(); vl=df[df["is_valid"]].copy()
    abl=tr[tr["has_abandoned"]]
    if len(abl)>0 and args.oversample_new>1:
        extra=pd.concat([abl]*(args.oversample_new-1),ignore_index=True)
        tr=pd.concat([tr,extra],ignore_index=True)
        print(f"🔥 Oversampled ×{args.oversample_new}: {len(abl)}→{len(tr[tr['has_abandoned']])} abdn rows")
    return pd.concat([tr,vl],ignore_index=True)


# ══════════════════════════════════════════════════════
# 7. LOAD PRETRAINED ENCODER WEIGHTS
# ══════════════════════════════════════════════════════
ENCODER_PREFIXES = ("layers.0.",)

def load_pretrained_weights(learner, path):
    dev=next(learner.model.parameters()).device
    old=load_learner(path,cpu=True)
    os=old.model.state_dict(); ns=learner.model.state_dict()
    loaded=skip_dec=skip_mm=0
    for k,v in os.items():
        if not any(k.startswith(p) for p in ENCODER_PREFIXES): skip_dec+=1; continue
        if k in ns and ns[k].shape==v.shape: ns[k]=v.to(dev); loaded+=1
        else: print(f"  ↳ shape mismatch: {k}"); skip_mm+=1
    learner.model.load_state_dict(ns)
    print(f"✅ Encoder: {loaded} layers loaded, {skip_dec} decoder skipped, {skip_mm} mismatched")
    return learner


# ══════════════════════════════════════════════════════
# 8. LEARNING CURVES
# ══════════════════════════════════════════════════════
def plot_learning_curves(csv_path, out_path):
    if not csv_path.exists(): return
    df=pd.read_csv(csv_path); df=df[df['epoch']!='epoch']
    for c in ['epoch','train_loss','valid_loss','dice_generator','abdn_lead_sensitivity']:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    fig,ax1=plt.subplots(figsize=(10,6))
    ax1.plot(df['train_loss'],label='Train Loss',color='blue',linestyle='--')
    ax1.plot(df['valid_loss'],label='Valid Loss', color='blue')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss',color='blue')
    ax1.tick_params(axis='y',labelcolor='blue')
    ax2=ax1.twinx()
    if 'dice_generator'       in df.columns: ax2.plot(df['dice_generator'],      label='Dice(Generator)', color='green')
    if 'abdn_lead_sensitivity' in df.columns: ax2.plot(df['abdn_lead_sensitivity'],label='Sensitivity(AbdnLead)',color='red')
    ax2.set_ylabel('Score(0-1)'); ax2.set_ylim(0,1.1)
    l1,b1=ax1.get_legend_handles_labels(); l2,b2=ax2.get_legend_handles_labels()
    ax2.legend(l1+l2,b1+b2,loc='upper right')
    plt.title('Training History'); plt.grid(True,alpha=0.3); fig.tight_layout()
    plt.savefig(out_path,dpi=150); print(f"📈 Curves → {out_path}"); plt.close()


# ══════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════
def finetune(args):
    out=pathlib.Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    df=build_dataframe(args)
    nva=df[df["is_valid"]&df["has_abandoned"]].shape[0]
    ntr=df[~df["is_valid"]&df["has_abandoned"]].shape[0]
    print(f"🔍 Train abdn:{ntr}  Val abdn:{nva}")
    if nva==0: print("⚠️  No abandoned lead in val — sensitivity will be nan")
    summarize_dataset(df)

    # Auto class weights
    class_weights=compute_class_weights(df,max_w=args.max_class_weight)
    print(f"⚖️  Weights: {[round(w,2) for w in class_weights.tolist()]}")

    # Save config
    import csv; from datetime import datetime
    csv_path=out/"training_history.csv"
    with open(csv_path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["# TRAINING CONFIG",datetime.now().strftime("%Y-%m-%d %H:%M")])
        for k,v in {"patch_size":args.patch_size,"std_size":args.std_size,
                    "lead_sample_ratio":args.lead_sample_ratio,
                    "max_class_weight":args.max_class_weight,
                    "class_weights":[round(x,2) for x in class_weights.tolist()],
                    "batch_size":args.batch_size,"epochs_decoder":args.epochs_decoder,
                    "epochs_head":args.epochs_head,"epochs_full":args.epochs_full,
                    "loss":"FocalLossFlat","backbone":"resnet50","grad_accum":4,"fp16":True,
                    "lr_p0":"2e-3","lr_p1":"6e-4","lr_p2":"slice(2e-6,2e-4)",
                    "NOTE":"PIL-level crop — no collation error"}.items():
            w.writerow([f"# {k}",v])
        w.writerow([])

    # Normalization
    stats_mean=[0.4945,0.4945,0.4945]; stats_std=[0.2267,0.2267,0.2267]

    # ── DataBlock ────────────────────────────────────────────────────────
    # RandomSplitCrop runs at PIL level (item_tfms) before ToTensor.
    # It resizes every image to std_size first (preserving lead thickness),
    # then crops patch_size × patch_size.
    # split_idx is set automatically by FastAI for train/val.
    dblock=DataBlock(
        blocks   =(ImageBlock,MaskBlock(codes=CLASS_NAMES)),
        get_x    =get_x, get_y=get_y,
        splitter =ColSplitter(col='is_valid'),
        item_tfms=[
            RandomSplitCrop(
                patch_size        =args.patch_size,
                std_size          =args.std_size,
                lead_sample_ratio =args.lead_sample_ratio,
                lead_classes      =(2,3),
            ),
        ],
        batch_tfms=[
            *aug_transforms(do_flip=True,flip_vert=False,max_rotate=15,
                            min_zoom=0.9,max_zoom=1.15,max_lighting=0.1,
                            max_warp=0.0,p_affine=0.75,p_lighting=0.5),
            Normalize.from_stats(stats_mean,stats_std),
        ],
    )
    dls=dblock.dataloaders(df,bs=args.batch_size,
                           num_workers=0,pin_memory=True,persistent_workers=False)

    print("\n📸 Batch inspection …")
    show_batch_inspection(dls,n=4,save_path=str(out/"batch_inspection.png"),
                          stats_mean=stats_mean,stats_std=stats_std)

    # ── Model ────────────────────────────────────────────────────────────
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func=FocalLossFlat(axis=1,weight=class_weights.to(device))
    learner=unet_learner(dls,resnet50,n_out=N_CLASSES,loss_func=loss_func,
                         metrics=[dice_generator,abdn_lead_sensitivity],
                         cbs=[CSVLogger(fname=str(out/"training_history.csv"),append=True),
                              GradientAccumulation(n_acc=4)]).to_fp16()

    print("\n📦 Loading encoder weights …")
    learner=load_pretrained_weights(learner,args.model_path)
    print(f"patch={args.patch_size} std={args.std_size} ratio={args.lead_sample_ratio} max_w={args.max_class_weight}")

    # Phase 0: decoder warmup
    print("\n--- Phase 0: Decoder warmup ---")
    learner.freeze_to(1); learner.fit_one_cycle(args.epochs_decoder,2e-3)

    # Phase 1: head only
    print("\n--- Phase 1: Head only ---")
    learner.freeze(); learner.fit_one_cycle(args.epochs_head,6e-4)

    # Phase 2: full finetune
    print("\n--- Phase 2: Full finetune ---")
    learner.unfreeze(); learner.path=out; learner.model_dir=""
    learner.fit_one_cycle(args.epochs_full,lr_max=slice(2e-6,2e-4),
                          cbs=SaveModelCallback(monitor='dice_generator',
                                                fname='best_seg',with_opt=False))

    print("🎨 Saving predictions …")
    learner.show_results(max_n=4,vmin=0,vmax=3)
    plt.savefig(out/"quick_peek.png"); plt.close()

    if (out/"best_seg.pth").exists():
        learner.load(str(out/"best_seg")); print("🏆 Loaded best checkpoint.")

    torch.save(learner.model.state_dict(), out/"seg_abdnL_weights.pth")
    print(f"\n✅ Weights → {out/'seg_abdnL_weights.pth'}")

    plot_learning_curves(out/"training_history.csv", out/"learning_curves.png")


# ══════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model_path",        default="C:/CIEDID_data/pkl/segmentation.pkl")
    p.add_argument("--new_imgs",          default="C:/CIEDID_data/AbdnL/data")
    p.add_argument("--new_masks",         default="C:/CIEDID_data/AbdnL/mask")
    p.add_argument("--output_dir",        default="C:/CIEDID_data/AbdnL/models")
    p.add_argument("--patch_size",        type=int,   default=512,
                   help="Crop size (px). GPU VRAM guide: 512→bs=1, 384→bs=2")
    p.add_argument("--std_size",          type=int,   default=1024,
                   help="Resize full-res image to this before cropping. "
                        "Keep >= patch_size. 1024 preserves lead thickness well.")
    p.add_argument("--lead_sample_ratio", type=float, default=0.7,
                   help="Fraction of train patches centered on lead pixels. "
                        "0=random, 1=always on lead.")
    p.add_argument("--max_class_weight",  type=float, default=50.0,
                   help="Cap for auto weights. Lower if generator Dice collapses.")
    p.add_argument("--epochs_decoder",    type=int,   default=5)
    p.add_argument("--epochs_head",       type=int,   default=5)
    p.add_argument("--epochs_full",       type=int,   default=10)
    p.add_argument("--batch_size",        type=int,   default=1)
    p.add_argument("--valid_split",       type=float, default=0.2)
    p.add_argument("--oversample_new",    type=int,   default=3)
    args=p.parse_args()
    args.new_imgs=pathlib.Path(args.new_imgs)
    args.new_masks=pathlib.Path(args.new_masks)
    args.model_path=pathlib.Path(args.model_path)
    finetune(args)