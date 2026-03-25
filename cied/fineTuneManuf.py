from fastai.vision.all import *
import pandas as pd
from pathlib import Path

# 1. Setup paths
abs_dataset_path = Path("/workdir/cied/Dataset")
new_data_file = abs_dataset_path / "your_new_finetune_data.xlsx"
exp_name = "s256_cls_h"

# 2. Load the saved learner (this preserves the original classes, head weights, and vocab)
learner = load_learner(f"models/{exp_name}.pkl")

# 3. Load new data
new_df = pd.read_excel(new_data_file)
new_df.loc[:, "patch_fname"] = new_df.loc[:, "patch_fname"].apply(
    lambda x: str(abs_dataset_path / "Classification" / x)
)

# 4. Verify the new data only contains classes the model already knows
original_classes = learner.dls.vocab
new_classes = set(new_df["Hersteller"].unique())
unknown_classes = new_classes - set(original_classes)
if unknown_classes:
    raise ValueError(f"New data contains unknown classes: {unknown_classes}. "
                     "Use the new-classes workflow instead.")

# 5. Build new DataLoaders using the SAME vocab as the original model
# Passing vocab= ensures label encoding is identical to what the model was trained on
new_dls = ImageDataLoaders.from_df(
    new_df,
    fn_col="patch_fname",
    label_col="Hersteller",
    valid_pct=0.2,
    item_tfms=Resize(256),
    batch_tfms=[*aug_transforms(size=256, min_scale=0.1)],
    bs=32,
    vocab=original_classes  # critical: keeps class index mapping identical
)

# 6. Swap in the new dataloader
learner.dls = new_dls

# 7. Find a good learning rate before committing to training
learner.lr_find()
# Inspect the plot, then set base_lr to the value just before the loss bottoms out
# A safe default for continuing training is around 1e-4 to 2e-4

# 8. Fine-tune
learner.fine_tune(epochs=4, base_lr=2e-4)

# 9. Save the updated model
new_exp_name = "s256_cls_h_finetuned"
learner.export(f"models/{new_exp_name}.pkl")