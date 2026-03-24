from fastai.vision.all import *
import pandas as pd
from pathlib import Path

# 1. Setup paths (Adjust these to your local environment)
abs_dataset_path = Path("/workdir/cied/Dataset")
new_data_file = abs_dataset_path / "your_new_finetune_data.xlsx"

# 2. Load the new spreadsheet
new_df = pd.read_excel(new_data_file)

# 3. Update paths so the computer can find the images
# Assuming images are in the same relative "Classification" folder as the original notebook
new_df.loc[:, "patch_fname"] = new_df.loc[:, "patch_fname"].apply(
    lambda x: str(abs_dataset_path / "Classification" / x)
)

# 4. Create a new DataLoader for the fine-tuning data
new_dls = ImageDataLoaders.from_df(
    new_df,
    fn_col="patch_fname",
    label_col="Hersteller",
    valid_pct=0.2, # Use 20% of the new data for validation
    item_tfms=Resize(256),
    batch_tfms=[*aug_transforms(size=256, min_scale=0.1)],
    bs=32 # Reduced batch size often helps with fine-tuning stability
)

# 5. Initialize the learner with the NEW data
# Use the same architecture (resnet50) as the original
learner = vision_learner(new_dls, resnet50, metrics=accuracy).to_fp16()

# 6. Load your saved .pkl model (the experiment name from your notebook was 's256_cls_h')
# Note: 'load_model' looks in the 'models/' directory relative to your notebook
exp_name = "s256_cls_h"
load_model(f"models/{exp_name}", learner.model, learner.opt)

# 7. Fine-tune! 
# 'fine_tune' is better than 'fit_one_cycle' for existing models.
# It freezes the body, trains the head for 1 epoch, then unfreezes everything and trains for 3.
learner.fine_tune(epochs=4, base_lr=1e-3)

# 8. Save your new fine-tuned version
new_exp_name = "s256_cls_h_finetuned"
save_model(f"models/{new_exp_name}", learner.model, learner.opt)