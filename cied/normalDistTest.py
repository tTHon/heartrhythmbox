import numpy as np
import pandas as pd
from scipy import stats


def analyze_data(file_path):
    try:
        # 1. โหลดไฟล์ CSV
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: ไม่สามารถอ่านไฟล์ได้ เนื่องจาก {e}")
        return

    # ปรับชื่อคอลัมน์ให้เป็นตัวพิมพ์เล็กชั่วคราวเพื่อป้องกันปัญหาพิมพ์ผิด (เช่น Height, weight, WEIGHT)
    df.columns = [col.strip().lower() for col in df.columns]

    # 2. ตรวจสอบว่ามีคอลัมน์ cm และ Kg หรือไม่
    if "kg" in df.columns and "cm" in df.columns:
        # แปลงข้อมูลเป็นตัวเลข (หากมี string ปนจะเปลี่ยนเป็น NaN)
        df["cm"] = pd.to_numeric(df["cm"], errors="coerce")
        df["kg"] = pd.to_numeric(df["kg"], errors="coerce")

        # สูตร BMI = น้ำหนัก (kg) / [ส่วนสูง (m)]^2
        # โค้ดนี้จะเช็คก่อนว่าส่วนสูงในไฟล์เป็นเซนติเมตรหรือไม่ (ถ้าส่วนใหญ่เกิน 3 ถือว่าเป็น cm ให้หาร 100)
        if df["cm"].mean() > 3:
            height_m = df["cm"] / 100
        else:
            height_m = df["cm"]

        # คำนวณ BMI และเพิ่มเข้าไปใน DataFrame เป็น Variable ใหม่
        df["bmi"] = df["kg"] / (height_m**2)
        print("-> คำนวณค่า BMI สำเร็จและเพิ่มเข้าสู่การวิเคราะห์แล้ว\n")
    else:
        print(
            "⚠️ คำเตือน: ไม่พบตัวแปร 'cm' หรือ 'kg' ในไฟล์ CSV (จะวิเคราะห์เฉพาะตัวแปรที่มี)"
        )

    print("=== รายงานผลการวิเคราะห์ตัวแปร (Variables Report) ===\n")

    # 3. วนลูปตรวจเช็คทุกตัวแปรใน DataFrame (รวม BMI ที่คำนวณได้ด้วย)
    for variable_name in df.columns:

        # ตรวจสอบว่าเป็นตัวแปรประเภทตัวเลขหรือไม่
        if pd.api.types.is_numeric_dtype(df[variable_name]):
            # ตัดค่าว่าง (NaN) ออกเพื่อความแม่นยำ
            data = df[variable_name].dropna()

            if len(data) < 3:
                print(
                    f"Variable Name: ** {variable_name.upper()} ** -> ข้อมูลมีน้อยเกินไป ไม่สามารถวิเคราะห์ได้"
                )
                print("-" * 50)
                continue

            # คำนวณค่าสถิติตามโจทย์
            mean_val = np.mean(data)
            sd_val = np.std(data, ddof=1)  # Sample SD
            median_val = np.median(data)
            iqr1 = np.percentile(data, 25)  # IQR 1 (Q1)
            iqr3 = np.percentile(data, 75)  # IQR 3 (Q3)

            # ทดสอบการแจกแจงแบบปกติ (Shapiro-Wilk Test)
            stat, p_value = stats.shapiro(data)
            alpha = 0.05
            is_normal = p_value > alpha

            # แสดงรายงานผลลัพธ์ระบุชื่อตัวแปรชัดเจน
            print(f"Variable Name: ** {variable_name.upper()} **")
            print(f"  - Mean:     {mean_val:.4f}")
            print(f"  - SD:       {sd_val:.4f}")
            print(f"  - Median:   {median_val:.4f}")
            print(f"  - IQR 1:    {iqr1:.4f}")
            print(f"  - IQR 3:    {iqr3:.4f}")
            print(
                f"  - การแจกแจง: {'เป็น Normal Distribution' if is_normal else 'ไม่เป็น Normal Distribution'} (p-value = {p_value:.5f})"
            )
            print("-" * 50)
        else:
            print(
                f"Variable Name: ** {variable_name.upper()} ** -> (ข้ามเนื่องจากไม่ใช่ตัวเลข)"
            )
            print("-" * 50)


# --- วิธีใช้งาน ---
# เปลี่ยนชื่อไฟล์ให้ตรงกับไฟล์ของคุณ เช่น 'biometric_data.csv'
analyze_data('CIED/AbdnL/A_modelCSV.csv')