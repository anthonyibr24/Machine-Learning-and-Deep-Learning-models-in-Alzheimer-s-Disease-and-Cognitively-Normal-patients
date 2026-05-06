import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# 1) CSV PATHS
# =========================================================
adni_ad_csv = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\vol+vbm\OASIS\AD\neuromorphometrics_all_patients_with_TIV_AD_OASIS.csv"
adni_cn_csv = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\vol+vbm\OASIS\CN\neuromorphometrics_all_patients_with_TIV_CN_OASIS.csv"
oasis_ad_csv = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\vol+vbm\ADNI\AD\neuromorphometrics_all_patients_with_TIV_AD_ADNI.csv"
oasis_cn_csv = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\vol+vbm\ADNI\CN\neuromorphometrics_all_patients_with_TIV_CN_ADNI.csv"
# =========================================================
# 2) LOAD CSV FILES
# =========================================================
adni_ad = pd.read_csv(adni_ad_csv)
adni_cn = pd.read_csv(adni_cn_csv)
oasis_ad = pd.read_csv(oasis_ad_csv)
oasis_cn = pd.read_csv(oasis_cn_csv)

# =========================================================
# 3) ADD LABELS
#    AD = 1, CN = 0
# =========================================================
adni_ad["Label"] = 1
adni_cn["Label"] = 0
oasis_ad["Label"] = 1
oasis_cn["Label"] = 0

# =========================================================
# 4) COMBINE INTO FULL DATASETS
# =========================================================
adni_df = pd.concat([adni_ad, adni_cn], ignore_index=True)
oasis_df = pd.concat([oasis_ad, oasis_cn], ignore_index=True)

# =========================================================
# 5) OPTIONAL: DROP NON-FEATURE COLUMNS
#    Add/remove columns here if needed
# =========================================================
columns_to_drop = ["PatientID", "SubjectID", "ImageID", "Group"]  # only if they exist

adni_df = adni_df.drop(columns=[c for c in columns_to_drop if c in adni_df.columns], errors="ignore")
oasis_df = oasis_df.drop(columns=[c for c in columns_to_drop if c in oasis_df.columns], errors="ignore")

# =========================================================
# 6) FIND FEATURE COLUMNS
# =========================================================
label_col = "Label"
tiv_col = "TIV"   # change this if your TIV column has another name

if tiv_col not in adni_df.columns:
    raise ValueError(f"TIV column '{tiv_col}' not found in ADNI dataset.")

if tiv_col not in oasis_df.columns:
    raise ValueError(f"TIV column '{tiv_col}' not found in OASIS dataset.")

vol_cols = [c for c in adni_df.columns if c.endswith("_Volume")]
vbm_cols = [c for c in adni_df.columns if c.endswith("_VBM")]

print(f"Found {len(vol_cols)} volume columns")
print(f"Found {len(vbm_cols)} VBM columns")

# =========================================================
# 7) CREATE VBM RATIO FEATURES
# =========================================================
def add_vbm_ratio_features(df, vbm_columns, tiv_column):
    df = df.copy()
    ratio_cols = []

    for col in vbm_columns:
        new_col = col.replace("_VBM", "_VBM_RATIO")
        df[new_col] = df[col] / df[tiv_column]
        ratio_cols.append(new_col)

    return df, ratio_cols

adni_df, vbm_ratio_cols = add_vbm_ratio_features(adni_df, vbm_cols, tiv_col)
oasis_df, _ = add_vbm_ratio_features(oasis_df, vbm_cols, tiv_col)

# =========================================================
# 8) DEFINE INPUT SETS
# =========================================================
feature_sets = {
    "VBM+VOL": vbm_cols + vol_cols,
    "VBM RATIO+VOL": vbm_ratio_cols + vol_cols,
    "VOL": vol_cols
}

# Keep only shared columns between ADNI and OASIS
for key in feature_sets:
    feature_sets[key] = [
        c for c in feature_sets[key]
        if c in adni_df.columns and c in oasis_df.columns
    ]

# =========================================================
# 9) METRIC FUNCTIONS
# =========================================================
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def cn_precision_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def evaluate_model(y_true, y_pred, y_prob):
    results = {}
    results["Accuracy"] = accuracy_score(y_true, y_pred)
    results["Precision"] = precision_score(y_true, y_pred, zero_division=0)   # AD Precision
    results["Recall"] = recall_score(y_true, y_pred, zero_division=0)         # AD Recall
    results["F1 Score"] = f1_score(y_true, y_pred, zero_division=0)
    results["Specificity"] = specificity_score(y_true, y_pred)                # CN Recall
    results["ROC-AUC"] = roc_auc_score(y_true, y_prob)

    results["AD Precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    results["AD Recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    results["CN Precision"] = cn_precision_score(y_true, y_pred)
    results["CN Recall"] = specificity_score(y_true, y_pred)

    return results

# =========================================================
# 10) MODELS
# =========================================================
models = {
    "RF": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "LR": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# =========================================================
# 11) TRAIN ON ADNI, TEST ON OASIS
# =========================================================
all_results = []

for input_name, feature_cols in feature_sets.items():
    print("\n" + "=" * 80)
    print(f"INPUT SET: {input_name}")
    print("=" * 80)

    X_train = adni_df[feature_cols].copy()
    y_train = adni_df[label_col].copy()

    X_test = oasis_df[feature_cols].copy()
    y_test = oasis_df[label_col].copy()

    print(f"ADNI train shape : {X_train.shape}")
    print(f"OASIS test shape : {X_test.shape}")

    for model_name, model in models.items():
        print("\n" + "#" * 70)
        print(f"{model_name}: {input_name} | Train on ADNI -> Test on OASIS")
        print("#" * 70)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results = evaluate_model(y_test, y_pred, y_prob)

        print(f"Accuracy    : {results['Accuracy']:.4f}")
        print(f"Precision   : {results['Precision']:.4f}")
        print(f"Recall      : {results['Recall']:.4f}")
        print(f"F1 Score    : {results['F1 Score']:.4f}")
        print(f"Specificity : {results['Specificity']:.4f}")
        print(f"ROC-AUC     : {results['ROC-AUC']:.4f}")
        print()
        print(f"AD Precision: {results['AD Precision']:.4f}")
        print(f"AD Recall   : {results['AD Recall']:.4f}")
        print(f"CN Precision: {results['CN Precision']:.4f}")
        print(f"CN Recall   : {results['CN Recall']:.4f}")

        row = {
            "Model": model_name,
            "Input": input_name,
            **results
        }
        all_results.append(row)

# =========================================================
# 12) SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(all_results)

save_path = r"C:\Users\antho\Desktop\ADNI_to_OASIS_external_validation_results.csv"
results_df.to_csv(save_path, index=False)

print("\n" + "=" * 80)
print("FINAL RESULTS TABLE")
print("=" * 80)
print(results_df)

print(f"\nSaved results to:\n{save_path}")