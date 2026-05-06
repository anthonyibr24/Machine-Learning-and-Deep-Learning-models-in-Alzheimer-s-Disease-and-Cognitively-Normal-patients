import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) LOAD DATA
# =========================
ad_df = pd.read_csv(r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\vol+vbm\ADNI\AD\neuromorphometrics_all_patients_with_TIV_AD_ADNI.csv")
cn_df = pd.read_csv(r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\vol+vbm\ADNI\CN\neuromorphometrics_all_patients_with_TIV_CN_ADNI.csv")

ad_df["Group"] = "AD"
cn_df["Group"] = "CN"

df = pd.concat([ad_df, cn_df], ignore_index=True)

# =========================
# 2) OUTPUT FOLDER
# =========================
output_dir = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Correlation_Maps"
os.makedirs(output_dir, exist_ok=True)

print("Files will be saved in:")
print(output_dir)

# =========================
# 3) SELECT ONLY VOLUME FEATURES
# =========================
volume_cols = [col for col in df.columns if col.endswith("_Volume")]

exclude_cols = {"TIV"}
volume_cols = [c for c in volume_cols if c not in exclude_cols]

print(f"Number of raw volume columns: {len(volume_cols)}")

# =========================
# 4) NORMALIZE BY TIV
# =========================
for col in volume_cols:
    df[col] = df[col] / df["TIV"]

print("Volume normalization by TIV completed.")

# =========================
# 5) MERGE LEFT + RIGHT
# =========================
merged_dict = {}

for col in volume_cols:
    base = col

    if base.startswith("Left_"):
        base = base.replace("Left_", "", 1)
    elif base.startswith("Right_"):
        base = base.replace("Right_", "", 1)

    base = base.replace("_Volume", "")

    if base not in merged_dict:
        merged_dict[base] = []

    merged_dict[base].append(col)

df_lr = pd.DataFrame(index=df.index)

for region, cols in merged_dict.items():
    df_lr[region] = df[cols].sum(axis=1)

df_lr["Group"] = df["Group"]

print(f"Number of merged left/right regions: {len(df_lr.columns) - 1}")

# =========================
# 6) DEFINE ASPECTS
# =========================
aspect_groups = {
    "Frontal": [
        "AOrG_anterior_orbital_gyrus",
        "FO_frontal_operculum",
        "FRP_frontal_pole",
        "GRe_gyrus_rectus",
        "LOrG_lateral_orbital_gyrus",
        "MFC_medial_frontal_cortex",
        "MFG_middle_frontal_gyrus",
        "MOrG_medial_orbital_gyrus",
        "MSFG_superior_frontal_gyrus_medial_segment",
        "OpIFG_opercular_part_of_the_inferior_frontal_gyrus",
        "OrIFG_orbital_part_of_the_inferior_frontal_gyrus",
        "POrG_posterior_orbital_gyrus",
        "SFG_superior_frontal_gyrus",
        "TrIFG_triangular_part_of_the_inferior_frontal_gyrus",
        "SCA_subcallosal_area"
    ],

    "Motor_Sensorimotor": [
        "MPrG_precentral_gyrus_medial_segment",
        "PrG_precentral_gyrus",
        "MPoG_postcentral_gyrus_medial_segment",
        "PoG_postcentral_gyrus",
        "SMC_supplementary_motor_cortex"
    ],

    "Temporal_Lateral": [
        "ITG_inferior_temporal_gyrus",
        "MTG_middle_temporal_gyrus",
        "STG_superior_temporal_gyrus",
        "TMP_temporal_pole",
        "TTG_transverse_temporal_gyrus",
        "PP_planum_polare",
        "PT_planum_temporale"
    ],

    "Medial_Temporal": [
        "Hippocampus",
        "Amygdala",
        "Ent_entorhinal_area",
        "PHG_parahippocampal_gyrus"
    ],

    "Parietal": [
        "AnG_angular_gyrus",
        "PCu_precuneus",
        "SPL_superior_parietal_lobule",
        "SMG_supramarginal_gyrus",
        "PO_parietal_operculum"
    ],

    "Occipital": [
        "Calc_calcarine_cortex",
        "Cun_cuneus",
        "IOG_inferior_occipital_gyrus",
        "MOG_middle_occipital_gyrus",
        "OCP_occipital_pole",
        "SOG_superior_occipital_gyrus"
    ],

    "Fusiform_Lingual": [
        "FuG_fusiform_gyrus",
        "LiG_lingual_gyrus",
        "OFuG_occipital_fusiform_gyrus"
    ],

    "Cingulate_Anterior": [
        "ACgG_anterior_cingulate_gyrus"
    ],

    "Cingulate_Middle": [
        "MCgG_middle_cingulate_gyrus"
    ],

    "Cingulate_Posterior": [
        "PCgG_posterior_cingulate_gyrus"
    ],

    "Insula": [
        "AIns_anterior_insula",
        "PIns_posterior_insula",
        "CO_central_operculum"
    ],

    "Basal_Ganglia": [
        "Caudate",
        "Putamen",
        "Pallidum",
        "Accumbens_Area"
    ],

    "Thalamus_VentralDC": [
        "Thalamus_Proper",
        "Ventral_DC"
    ],

    "Basal_Forebrain": [
        "Basal_Forebrain",
        "Optic_Chiasm"
    ],

    "Cerebellum": [
        "Cerebellum_Exterior",
        "Cerebellum_White_Matter",
        "Cerebellar_Vermal_Lobules_I_V",
        "Cerebellar_Vermal_Lobules_VI_VII",
        "Cerebellar_Vermal_Lobules_VIII_X"
    ],

    "Brainstem": [
        "Brain_Stem"
    ],

    "Lateral_Ventricular_System": [
        "Lateral_Ventricle",
        "Inf_Lat_Vent"
    ],

    "Midline_Ventricles_CSF": [
        "3rd_Ventricle",
        "4th_Ventricle",
        "CSF"
    ],

    "White_Matter": [
        "Cerebral_White_Matter"
    ]
}

# =========================
# 7) BUILD ASPECT DATAFRAME
# =========================
aspect_df = pd.DataFrame(index=df.index)

print("\nAspect composition:")
for aspect, regions in aspect_groups.items():
    valid_regions = [r for r in regions if r in df_lr.columns]
    print(f"{aspect}: {len(valid_regions)} regions")

    if len(valid_regions) > 0:
        aspect_df[aspect] = df_lr[valid_regions].sum(axis=1)

aspect_df["Group"] = df_lr["Group"]
aspect_df = aspect_df.dropna(axis=1, how="all")

print("\nFinal aspect columns:")
print(aspect_df.columns.tolist())
print(f"\nNumber of aspects used: {aspect_df.drop(columns=['Group']).shape[1]}")

# =========================
# 8) CORRELATION MATRICES
# =========================
all_corr = aspect_df.drop(columns=["Group"]).corr(method="pearson")
ad_corr = aspect_df[aspect_df["Group"] == "AD"].drop(columns=["Group"]).corr(method="pearson")
cn_corr = aspect_df[aspect_df["Group"] == "CN"].drop(columns=["Group"]).corr(method="pearson")
diff_corr = ad_corr - cn_corr

# =========================
# 9) HIGHLIGHT THRESHOLDS
# =========================
corr_threshold = 0.50      # highlight if |r| >= 0.50
diff_threshold = 0.20      # highlight if |AD - CN| >= 0.20

# =========================
# 10) PLOT + SAVE FUNCTION
# =========================
def plot_corr_matrix(
    corr_matrix,
    title,
    filename,
    vmin=-1,
    vmax=1,
    highlight_threshold=0.50,
    is_difference_map=False
):
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))

    ax.set_xticklabels(
        corr_matrix.columns,
        rotation=45,
        ha="right",
        fontsize=9
    )

    ax.set_yticklabels(
        corr_matrix.index,
        fontsize=9
    )

    # Write all values + highlight important squares
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]

            if pd.isna(val):
                continue

            # show all correlation/difference values
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

            # do not highlight diagonal for normal correlation maps
            if not is_difference_map and i == j:
                continue

            # highlight important squares only
            if abs(val) >= highlight_threshold:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5
                )
                ax.add_patch(rect)

    ax.set_title(title, fontsize=16)
    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {full_path}")

    plt.show()
    plt.close(fig)

# =========================
# 11) SAVE ALL MAPS
# =========================
plot_corr_matrix(
    all_corr,
    f"Correlation Map (All Subjects) - Highlighted |r| ≥ {corr_threshold}",
    "corr_all_highlighted.pdf",
    vmin=-1,
    vmax=1,
    highlight_threshold=corr_threshold,
    is_difference_map=False
)

plot_corr_matrix(
    ad_corr,
    f"Correlation Map (AD) - Highlighted |r| ≥ {corr_threshold}",
    "corr_ad_highlighted.pdf",
    vmin=-1,
    vmax=1,
    highlight_threshold=corr_threshold,
    is_difference_map=False
)

plot_corr_matrix(
    cn_corr,
    f"Correlation Map (CN) - Highlighted |r| ≥ {corr_threshold}",
    "corr_cn_highlighted.pdf",
    vmin=-1,
    vmax=1,
    highlight_threshold=corr_threshold,
    is_difference_map=False
)

plot_corr_matrix(
    diff_corr,
    f"Correlation Difference Map (AD - CN) - Highlighted |Δr| ≥ {diff_threshold}",
    "corr_diff_highlighted.pdf",
    vmin=-0.5,
    vmax=0.5,
    highlight_threshold=diff_threshold,
    is_difference_map=True
)