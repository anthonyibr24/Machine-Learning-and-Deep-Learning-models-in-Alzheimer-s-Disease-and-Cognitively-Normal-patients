import os
import copy
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity
from monai.networks.nets import resnet18


# =========================================================
# 1. PATHS
# =========================================================
ad_folder = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\mwp1_flat_dataset\AD"
cn_folder = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\mwp1_flat_dataset\CN"


# =========================================================
# 2. LOAD ALL FILES
# AD = 1
# CN = 0
# =========================================================
image_paths = []
labels = []
patient_ids = []

def extract_patient_id(filename):
    base = filename.replace(".nii.gz", "").replace(".nii", "")
    parts = base.split("_")
    return "_".join(parts[1:4])

for f in os.listdir(ad_folder):
    if f.endswith(".nii") or f.endswith(".nii.gz"):
        image_paths.append(os.path.join(ad_folder, f))
        labels.append(1)
        patient_ids.append(extract_patient_id(f))

for f in os.listdir(cn_folder):
    if f.endswith(".nii") or f.endswith(".nii.gz"):
        image_paths.append(os.path.join(cn_folder, f))
        labels.append(0)
        patient_ids.append(extract_patient_id(f))

image_paths = np.array(image_paths)
labels = np.array(labels)
patient_ids = np.array(patient_ids)

print("Total images found:", len(image_paths))
print("AD images:", np.sum(labels == 1))
print("CN images:", np.sum(labels == 0))
print("Unique patients:", len(np.unique(patient_ids)))


# =========================================================
# 3. TRANSFORMS
# =========================================================
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((96, 96, 96))
])


# =========================================================
# 4. DATASET
# =========================================================
class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transforms):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.transforms(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# =========================================================
# 5. DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 6. CROSS VALIDATION SETTINGS
# =========================================================
num_folds = 4
num_epochs = 15
batch_size = 2
learning_rate = 1e-4
polyak_k = 5

skf = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)

all_fold_results = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def evaluate_model(model, loader, loss_function, device):
    model.eval()
    val_loss_total = 0.0

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            inputs = batch_data.to(device)
            targets = batch_labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss_total += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    avg_val_loss = val_loss_total / len(loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2

    # ---------------------------
    # NEW: PER-CLASS METRICS
    # AD = 1, CN = 0
    # ---------------------------
    ad_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    ad_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    cn_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    cn_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0

    return {
        "val_loss": avg_val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "roc_auc": roc_auc,
        "cm": cm,
        "ad_precision": ad_precision,
        "ad_recall": ad_recall,
        "cn_precision": cn_precision,
        "cn_recall": cn_recall
    }


def average_checkpoints(state_dict_list):
    avg_state = copy.deepcopy(state_dict_list[0])

    for key in avg_state.keys():
        for i in range(1, len(state_dict_list)):
            avg_state[key] += state_dict_list[i][key]
        avg_state[key] = avg_state[key] / len(state_dict_list)

    return avg_state


# =========================================================
# 7. CROSS VALIDATION LOOP
# =========================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels, groups=patient_ids), start=1):
    print("\n" + "=" * 70)
    print(f"FOLD {fold}/{num_folds}")
    print("=" * 70)

    train_images = image_paths[train_idx]
    train_labels = labels[train_idx]
    train_patients = patient_ids[train_idx]

    val_images = image_paths[val_idx]
    val_labels = labels[val_idx]
    val_patients = patient_ids[val_idx]

    print("Train size:", len(train_images))
    print("Val size  :", len(val_images))
    print("Train AD  :", np.sum(train_labels == 1))
    print("Train CN  :", np.sum(train_labels == 0))
    print("Val AD    :", np.sum(val_labels == 1))
    print("Val CN    :", np.sum(val_labels == 0))
    print("Train patients:", len(np.unique(train_patients)))
    print("Val patients  :", len(np.unique(val_patients)))

    overlap = set(train_patients).intersection(set(val_patients))
    print("Patient overlap between train and val:", len(overlap))

    train_dataset = MRIDataset(train_images, train_labels, transforms)
    val_dataset = MRIDataset(val_images, val_labels, transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = resnet18(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # =====================================================
    # CLASS-WEIGHTED LOSS
    # =====================================================
    num_ad = np.sum(train_labels == 1)
    num_cn = np.sum(train_labels == 0)

    class_weights = torch.tensor(
        [len(train_labels) / (2 * num_cn), len(train_labels) / (2 * num_ad)],
        dtype=torch.float32
    ).to(device)

    print("Class weights:", class_weights.cpu().numpy())

    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    last_k_states = []

    # -----------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for batch_data, batch_labels in train_loader:
            inputs = batch_data.to(device)
            targets = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        last_k_states.append(copy.deepcopy(model.state_dict()))
        if len(last_k_states) > polyak_k:
            last_k_states.pop(0)

        epoch_metrics = evaluate_model(model, val_loader, loss_function, device)

        print(f"\nFold {fold} - Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss  : {avg_train_loss:.4f}")
        print(f"Val Loss    : {epoch_metrics['val_loss']:.4f}")
        print(f"Accuracy    : {epoch_metrics['accuracy']:.4f}")
        print(f"Precision   : {epoch_metrics['precision']:.4f}")
        print(f"Recall      : {epoch_metrics['recall']:.4f}")
        print(f"F1 Score    : {epoch_metrics['f1']:.4f}")
        print(f"Specificity : {epoch_metrics['specificity']:.4f}")
        print(f"Balanced Acc: {epoch_metrics['balanced_acc']:.4f}")
        print(f"ROC-AUC     : {epoch_metrics['roc_auc']:.4f}")
        print(f"AD Precision: {epoch_metrics['ad_precision']:.4f}")
        print(f"AD Recall   : {epoch_metrics['ad_recall']:.4f}")
        print(f"CN Precision: {epoch_metrics['cn_precision']:.4f}")
        print(f"CN Recall   : {epoch_metrics['cn_recall']:.4f}")
        print("Confusion Matrix:")
        print(epoch_metrics["cm"])

    # -----------------------------------------------------
    # Final model = average last k epochs
    # -----------------------------------------------------
    avg_state = average_checkpoints(last_k_states)
    model.load_state_dict(avg_state)

    final_metrics = evaluate_model(model, val_loader, loss_function, device)

    print("\n" + "-" * 70)
    print(f"Fold {fold} FINAL METRICS AFTER AVERAGING LAST {len(last_k_states)} EPOCHS")
    print("-" * 70)
    print(f"Accuracy    : {final_metrics['accuracy']:.4f}")
    print(f"Precision   : {final_metrics['precision']:.4f}")
    print(f"Recall      : {final_metrics['recall']:.4f}")
    print(f"F1 Score    : {final_metrics['f1']:.4f}")
    print(f"Specificity : {final_metrics['specificity']:.4f}")
    print(f"Balanced Acc: {final_metrics['balanced_acc']:.4f}")
    print(f"ROC-AUC     : {final_metrics['roc_auc']:.4f}")
    print(f"AD Precision: {final_metrics['ad_precision']:.4f}")
    print(f"AD Recall   : {final_metrics['ad_recall']:.4f}")
    print(f"CN Precision: {final_metrics['cn_precision']:.4f}")
    print(f"CN Recall   : {final_metrics['cn_recall']:.4f}")
    print("Confusion Matrix:")
    print(final_metrics["cm"])

    all_fold_results.append({
        "accuracy": final_metrics["accuracy"],
        "precision": final_metrics["precision"],
        "recall": final_metrics["recall"],
        "f1": final_metrics["f1"],
        "specificity": final_metrics["specificity"],
        "balanced_acc": final_metrics["balanced_acc"],
        "roc_auc": final_metrics["roc_auc"],
        "ad_precision": final_metrics["ad_precision"],
        "ad_recall": final_metrics["ad_recall"],
        "cn_precision": final_metrics["cn_precision"],
        "cn_recall": final_metrics["cn_recall"]
    })


# =========================================================
# 8. FINAL AVERAGE RESULTS
# =========================================================
accs = [r["accuracy"] for r in all_fold_results]
precs = [r["precision"] for r in all_fold_results]
recs = [r["recall"] for r in all_fold_results]
f1s = [r["f1"] for r in all_fold_results]
specs = [r["specificity"] for r in all_fold_results]
baccs = [r["balanced_acc"] for r in all_fold_results]
aucs = [r["roc_auc"] for r in all_fold_results]

ad_precs = [r["ad_precision"] for r in all_fold_results]
ad_recs = [r["ad_recall"] for r in all_fold_results]
cn_precs = [r["cn_precision"] for r in all_fold_results]
cn_recs = [r["cn_recall"] for r in all_fold_results]

print("\n" + "#" * 70)
print("FINAL 4-FOLD CROSS-VALIDATION RESULTS")
print("#" * 70)
print(f"Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision   : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall      : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1 Score    : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Specificity : {np.mean(specs):.4f} ± {np.std(specs):.4f}")
print(f"Balanced Acc: {np.mean(baccs):.4f} ± {np.std(baccs):.4f}")
print(f"ROC-AUC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"AD Precision: {np.mean(ad_precs):.4f} ± {np.std(ad_precs):.4f}")
print(f"AD Recall   : {np.mean(ad_recs):.4f} ± {np.std(ad_recs):.4f}")
print(f"CN Precision: {np.mean(cn_precs):.4f} ± {np.std(cn_precs):.4f}")
print(f"CN Recall   : {np.mean(cn_recs):.4f} ± {np.std(cn_recs):.4f}")