from fastai.vision.all import *

# 1. Define the path to your NEW data and OLD model
abs_dataset_path = Path("/workdir/cied/Dataset")
new_data_path = abs_dataset_path / "your_new_model_data.xlsx"
old_model_pkl = "s256_cls_model.pkl"

# 2. Create DataLoaders for the NEW set of models
new_df = pd.read_excel(new_data_path)
# Ensure paths are absolute
new_df["patch_fname"] = new_df["patch_fname"].apply(lambda x: str(abs_dataset_path / "Classification" / x))

dls_new = ImageDataLoaders.from_df(
    new_df, 
    fn_col="patch_fname", 
    label_col="Exaktes_Modell", 
    valid_pct=0.2,
    item_tfms=Resize(256),
    batch_tfms=[*aug_transforms(size=256)]
)

# 3. Initialize a fresh ResNet50 learner
# This creates a brand new 'Head' matching your NEW number of classes
learner = vision_learner(dls_new, resnet50, metrics=accuracy).to_fp16()

# 4. Load the old specialized model
# We use load_learner to access the weights of the previous 29-class model
old_learner = load_learner(old_model_pkl)

# 5. THE SURGERY: Copy only the Backbone (Layer 0)
# This keeps the 'knowledge' of how to see pacemakers, but ignores the old labels
learner.model[0].load_state_dict(old_learner.model[0].state_dict())

# 6. Fine-tune
# We use a slightly lower learning rate because the backbone is already quite smart
learner.fine_tune(10, base_lr=2e-4)

# 7. Save the new version
learner.export("s256_cls_model_v2_updated.pkl")

# Create an Interpretation object
interp = ClassificationInterpretation.from_learner(learner)

# Plot the full matrix
# We increase figsize and dpi because you have many classes
interp.plot_confusion_matrix(figsize=(15,15), dpi=60)
# This shows you which pairs the model confuses most often
# e.g., 'Medtronic A' predicted as 'Medtronic B' 5 times
interp.most_confused(min_val=2)
# This plots the images with the highest 'loss'
# It shows: Prediction / Actual / Loss / Probability
interp.plot_top_losses(9, nrows=3)