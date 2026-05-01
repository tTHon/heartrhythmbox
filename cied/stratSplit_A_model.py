# Stratified train/test split for the CIED dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) load data
df = pd.read_csv("cied/AModelSplit.csv")

# 2) separate valid rows (with noActiveL) from NaN rows (without noActiveL)
df_valid = df[df["noActiveL"].notna()].copy()
df_nan   = df[df["noActiveL"].isna()].copy()

# 3) split
train_df, test_df = train_test_split(
    df_valid,
    test_size=0.2,
    stratify=df_valid["noActiveL"],
    random_state=42
)

# 👉 3) assign split label
df["split"] = "unused"  # default value for all rows
df.loc[train_df.index, "split"] = 0
df.loc[test_df.index, "split"] = 1

# 4) save
df.to_csv("cied/AModelSplit.csv", index=False)