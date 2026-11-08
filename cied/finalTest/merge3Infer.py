# Merge the three stage outputs into preds.csv for score_heldout.py
import pandas as pd
RESCUED = ["1612"]          # IDs whose crop was made/fixed by hand -> crop_status = manual (edit if different)
a = pd.read_csv("C:/CIEDID_data/finalTest/per_image_results.csv", dtype={"ID": str})[["ID", "prob_80"]].rename(columns={"prob_80": "abdn_px"})
m = pd.read_csv("C:/CIEDID_data/finalTest/pred_manuf.csv", dtype={"ID": str})[["ID", "manuf_pred"]]
c = pd.read_csv("C:/CIEDID_data/finalTest/pred_modelclass.csv", dtype={"ID": str})[["ID", "modelclass_pred"]]
out = a.merge(m, on="ID", validate="1:1").merge(c, on="ID", validate="1:1")
out["crop_status"] = out["ID"].isin(RESCUED).map({True: "manual", False: "auto"})
assert len(out) == 324, f"expected 324 rows, got {len(out)}"
out.to_csv("C:/CIEDID_data/finalTest/preds.csv", index=False)
print("preds.csv written:", len(out), "rows | manual crops:", int((out.crop_status == "manual").sum()))