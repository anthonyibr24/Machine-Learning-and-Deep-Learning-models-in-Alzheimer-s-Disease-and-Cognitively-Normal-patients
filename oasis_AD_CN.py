import os
import re
import shutil
import pandas as pd

# =========================================================
# SETTINGS
# =========================================================
MAX_DAY_DIFF = 180   # about 6 months

# =========================================================
# PATHS
# =========================================================
csv_path = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\OASIS3_UDSb4_cdr.csv"
source_dir = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\firuz-20260403_181213"
output_dir = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\first_valid_visit_dataset_6months"

# =========================================================
# READ CSV
# =========================================================
df = pd.read_csv(csv_path)

id_col = "OASISID"
label_col = "dx1"
days_col = "days_to_visit"

df[id_col] = df[id_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip()
df[days_col] = pd.to_numeric(df[days_col], errors="coerce")

# keep only rows with valid visit day
df = df.dropna(subset=[days_col])

# =========================================================
# FIND FIRST VALID AD/CN VISIT PER PATIENT
# Rule:
#   go visit by visit
#   first AD/CN visit decides label
# =========================================================
patient_target = {}

for pid, group in df.groupby(id_col):
    group = group.sort_values(days_col)

    chosen_label = None
    chosen_day = None

    for _, row in group.iterrows():
        dx = str(row[label_col]).strip()
        day = int(row[days_col])

        if dx == "AD Dementia":
            chosen_label = "AD"
            chosen_day = day
            break
        elif dx == "Cognitively normal":
            chosen_label = "CN"
            chosen_day = day
            break

    if chosen_label is not None:
        patient_target[pid] = {
            "label": chosen_label,
            "target_day": chosen_day
        }

print("Patients with first valid AD/CN visit:", len(patient_target))
print("AD patients:", sum(v["label"] == "AD" for v in patient_target.values()))
print("CN patients:", sum(v["label"] == "CN" for v in patient_target.values()))

# =========================================================
# FIND MRI SESSION FOLDERS
# Example: OAS30001_MR_d0129
# =========================================================
pattern = re.compile(r"^(OAS3\d+)_MR_d(\d+)$")

patient_mris = {}

for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)

    if not os.path.isdir(item_path):
        continue

    match = pattern.match(item)
    if not match:
        print(f"Skipped unexpected folder name: {item}")
        continue

    pid = match.group(1)
    mri_day = int(match.group(2))

    patient_mris.setdefault(pid, []).append((mri_day, item))

print("Patients with MRI folders:", len(patient_mris))

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================
ad_dir = os.path.join(output_dir, "AD")
cn_dir = os.path.join(output_dir, "CN")

os.makedirs(ad_dir, exist_ok=True)
os.makedirs(cn_dir, exist_ok=True)

# =========================================================
# FOR EACH PATIENT:
# choose MRI closest to first valid AD/CN visit
# but only keep if within 6 months (180 days)
# =========================================================
copied_ad = 0
copied_cn = 0
skipped_no_mri = 0
skipped_too_far = 0
selected_rows = []

for pid, info in patient_target.items():
    if pid not in patient_mris:
        skipped_no_mri += 1
        continue

    label = info["label"]
    target_day = info["target_day"]
    mri_list = patient_mris[pid]

    # choose MRI closest to chosen clinical day
    best_mri_day, best_folder = min(mri_list, key=lambda x: abs(x[0] - target_day))
    day_diff = abs(best_mri_day - target_day)

    # keep only if within 6 months
    if day_diff > MAX_DAY_DIFF:
        skipped_too_far += 1
        continue

    src = os.path.join(source_dir, best_folder)

    if label == "AD":
        dst = os.path.join(ad_dir, best_folder)
        copied_ad += 1
    else:
        dst = os.path.join(cn_dir, best_folder)
        copied_cn += 1

    shutil.copytree(src, dst, dirs_exist_ok=True)

    selected_rows.append({
        "patient_id": pid,
        "label": label,
        "chosen_clinical_day": target_day,
        "selected_mri_day": best_mri_day,
        "selected_folder": best_folder,
        "day_difference": day_diff,
        "num_mri_sessions_found": len(mri_list)
    })

    print(f"{best_folder} -> {label} (diff={day_diff} days)")

# =========================================================
# SAVE SUMMARY
# =========================================================
summary_df = pd.DataFrame(selected_rows)
summary_csv = os.path.join(output_dir, "selected_first_valid_visit_summary_6months.csv")
summary_df.to_csv(summary_csv, index=False)

# =========================================================
# EXTRA CHECKS
# =========================================================
print("\n" + "=" * 70)
print("DAY DIFFERENCE SUMMARY")
print("=" * 70)

if len(summary_df) > 0:
    print(summary_df["day_difference"].describe())
    print("\nExact matches      :", (summary_df["day_difference"] == 0).sum())
    print("Within 30 days     :", (summary_df["day_difference"] <= 30).sum())
    print("Within 90 days     :", (summary_df["day_difference"] <= 90).sum())
    print("Within 180 days    :", (summary_df["day_difference"] <= 180).sum())
else:
    print("No patients were selected.")

# =========================================================
# PRINT SUMMARY
# =========================================================
print("\n==================== DONE ====================")
print("AD copied               :", copied_ad)
print("CN copied               :", copied_cn)
print("Total copied            :", copied_ad + copied_cn)
print("Skipped (no MRI)        :", skipped_no_mri)
print("Skipped (too far >180d) :", skipped_too_far)
print("Summary CSV saved to    :", summary_csv)