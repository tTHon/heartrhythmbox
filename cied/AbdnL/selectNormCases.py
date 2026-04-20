# select_normal_cases.py
#
# Adds column "forAbdnModel" to workbook.csv:
#   1 = selected for abandoned-lead segmentation model training
#   0 = not selected
#
# Selection strategy (100 normal cases total):
#   70 cases WITH active transvenous lead
#      PPM dual   : 40
#      ICD single : 11
#      CRTD       :  9
#      ICD dual   :  6
#      PPM single :  3
#      CRTP       :  1
#
#   30 cases WITHOUT transvenous lead
#      Leadless   : 24  (Micra / LINQ)
#      subcut ICD :  6
#
# Abandoned cases (isAbandon==1) are NOT touched.
# Sampling is stratified by Type and reproducible (random_state=42).

import pathlib
import argparse
import pandas as pd

WITH_LEAD_TARGETS = {
    'PPM dual':   40,
    'ICD single': 11,
    'CRTD':        9,
    'ICD dual':    6,
    'PPM single':  3,
    'CRTP':        1,
}
NO_LEAD_TARGETS = {
    'Leadless':   24,
    'subcut ICD':  6,
}
NO_LEAD_TYPES = {'Leadless', 'subcut ICD'}


def select_cases(workbook_path, random_state=42):
    df = pd.read_csv(workbook_path)
    df['forAbdnModel'] = 0
    
    # ป้องกันเรื่อง Data Type: แปลงเป็นตัวเลขเพื่อให้เปรียบเทียบกับ 1 ได้แม่นยำ
    final_test_val = pd.to_numeric(df['FinalTest'], errors='coerce')

    eligible = df[
        (df['isAbandon'] == 0) & 
        (final_test_val != 1) &
        (df['whyExclude'].isna() | (df['whyExclude'].astype(str).str.strip() == ''))         
    ].copy()

    selected_ids = []

    # With-lead types
    has_lead = eligible[~eligible['Type'].isin(NO_LEAD_TYPES)]
    for type_name, n_needed in WITH_LEAD_TARGETS.items():
        pool = has_lead[has_lead['Type'] == type_name]
        if len(pool) < n_needed:
            print(f"  ⚠️  {type_name}: need {n_needed}, only {len(pool)} available — taking all")
            n_needed = len(pool)
        selected_ids.extend(pool.sample(n=n_needed, random_state=random_state)['ID'].tolist())

    # No-lead types
    no_lead = eligible[eligible['Type'].isin(NO_LEAD_TYPES)]
    for type_name, n_needed in NO_LEAD_TARGETS.items():
        pool = no_lead[no_lead['Type'] == type_name]
        if len(pool) < n_needed:
            print(f"  ⚠️  {type_name}: need {n_needed}, only {len(pool)} available — taking all")
            n_needed = len(pool)
        selected_ids.extend(pool.sample(n=n_needed, random_state=random_state)['ID'].tolist())

    df.loc[df['ID'].isin(selected_ids), 'forAbdnModel'] = 1
    return df


def print_summary(df):
    selected = df[df['forAbdnModel'] == 1]
    # นับจำนวน FinalTest = 1 ในกลุ่มที่ถูกเลือก (ควรจะเป็น 0)
    # ใช้ pd.to_numeric เพื่อความแม่นยำในการนับ
    final_test_count = (pd.to_numeric(selected['FinalTest'], errors='coerce') == 1).sum()
    # แสดงจำนวน FinalTest ที่หลุดมา
    print(f"  FinalTest = 1 count in selection: {final_test_count} cases")

    print(f"\n{'='*52}")
    print(f"  SELECTION SUMMARY  ({len(selected)} cases total)")
    print(f"{'='*52}")
    print(f"\n  By Type:")
    for t, n in selected['Type'].value_counts().items():
        group = "no-lead " if t in NO_LEAD_TYPES else "has-lead"
        print(f"    [{group}]  {t:15s}: {n:3d}")
    print(f"\n  By Manufacturer:")
    for m, n in selected['Manuf'].value_counts().items():
        print(f"    {m:15s}: {n:3d}")
    print(f"{'='*52}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook",     default="cied/workbook.csv")
    parser.add_argument("--output",       default=None,
                        help="Output path — defaults to overwriting input file")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    workbook_path = pathlib.Path(args.workbook)
    output_path   = pathlib.Path(args.output) if args.output else workbook_path

    if not workbook_path.exists():
        raise FileNotFoundError(f"Not found: {workbook_path}")

    print(f"📂 Reading : {workbook_path}")
    df = select_cases(workbook_path, random_state=args.random_state)
    print_summary(df)
    df.to_csv(output_path, index=False)
    print(f"💾 Saved   : {output_path}")
    print(f"   Column 'forAbdnModel' — 1=selected for training, 0=not selected")


if __name__ == "__main__":
    main()