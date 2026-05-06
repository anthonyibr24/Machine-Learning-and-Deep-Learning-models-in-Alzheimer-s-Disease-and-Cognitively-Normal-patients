import os
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity
from monai.networks.nets import DenseNet121

# =========================================================
# 1. PATHS
# =========================================================
adni_ad_folder = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\mwp1_flat_dataset\AD"
adni_cn_folder = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\Oasis\mwp1_flat_dataset\CN"

oasis_ad_folder = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\ADNI 895 Patients\895 patients 3D\mwp1_flat\AD"
oasis_cn_folder = r"C:\Users\antho\Desktop\CUD\Semester 6\ADNI\ADNI 895 Patients\895 patients 3D\mwp1_flat\CN"

# =========================================================
# 2. DATA LOADING & SUBJECT-LEVEL SPLIT
# =========================================================
def get_paths_and_ids(ad_folder, cn_folder):
    paths, labels, ids = [], [], []
    for folder, label in [(ad_folder, 1), (cn_folder, 0)]:
        for f in os.listdir(folder):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                paths.append(os.path.join(folder, f))
                labels.append(label)
                base = f.replace(".nii.gz", "").replace(".nii", "")
                subject_id = "_".join(base.split("_")[1:4])
                ids.append(subject_id)
    return np.array(paths), np.array(labels), np.array(ids)

adni_paths, adni_labels, adni_ids = get_paths_and_ids(adni_ad_folder, adni_cn_folder)
oasis_paths, oasis_labels, oasis_ids = get_paths_and_ids(oasis_ad_folder, oasis_cn_folder)

unique_patients = np.unique(adni_ids)
train_pts, val_pts = train_test_split(unique_patients, test_size=0.1, random_state=42)

train_mask = np.isin(adni_ids, train_pts)
val_mask = np.isin(adni_ids, val_pts)

train_paths, train_labels = adni_paths[train_mask], adni_labels[train_mask]
val_paths, val_labels = adni_paths[val_mask], adni_labels[val_mask]

# =========================================================
# 3. TRANSFORMS & DATASET
# =========================================================
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((96, 96, 96))
])

class MRIDataset(Dataset):
    def __init__(self, paths, labels, transforms):
        self.paths = paths
        self.labels = labels
        self.transforms = transforms
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = self.transforms(self.paths[idx])
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, lbl

train_loader = DataLoader(MRIDataset(train_paths, train_labels, transforms), batch_size=2, shuffle=True)
val_loader   = DataLoader(MRIDataset(val_paths, val_labels, transforms), batch_size=2, shuffle=False)
test_loader  = DataLoader(MRIDataset(oasis_paths, oasis_labels, transforms), batch_size=2, shuffle=False)

# =========================================================
# 4. ENHANCED EVALUATION FUNCTION
# =========================================================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            y_prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_true.extend(targets.numpy())
    
    # Per-Class Metrics [0: CN, 1: AD]
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    
    # Global Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate Sensitivity & Specificity explicitly
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_overall": precision_score(y_true, y_pred, zero_division=0),
        "recall_overall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "spec": specificity,
        "bal_acc": balanced_acc,
        "ad_recall": recalls[1],
        "ad_precision": precisions[1],
        "cn_recall": recalls[0],
        "cn_precision": precisions[0],
        "cm": cm
    }

# =========================================================
# 5. TRAINING SETUP
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

num_ad = np.sum(train_labels == 1)
num_cn = np.sum(train_labels == 0)
weights = torch.tensor([len(train_labels)/(2*num_cn), len(train_labels)/(2*num_ad)], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 15
polyak_k = 5
last_k_states = []

# =========================================================
# 6. TRAINING LOOP
# =========================================================
print("Starting Training on ADNI...")
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss_fn(model(inputs.to(device)), targets.to(device)).backward()
        optimizer.step()

    last_k_states.append(copy.deepcopy(model.state_dict()))
    if len(last_k_states) > polyak_k: last_k_states.pop(0)

    m = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {m['accuracy']:.4f} | Val F1: {m['f1']:.4f}")

# =========================================================
# 7. FINAL EXTERNAL EVALUATION (OASIS)
# =========================================================
print("\n" + "="*60)
print("FINAL EXTERNAL VALIDATION RESULTS (OASIS)")
print("="*60)

# Apply Polyak Average
avg_state = copy.deepcopy(last_k_states[0])
for key in avg_state.keys():
    for i in range(1, len(last_k_states)):
        avg_state[key] += last_k_states[i][key]
    avg_state[key] = avg_state[key] / len(last_k_states)
model.load_state_dict(avg_state)

# Evaluate
res = evaluate(model, test_loader, device)

# Matches previous table format exactly
print(f"Accuracy:    {res['accuracy']:.4f}")
print(f"Precision:   {res['precision_overall']:.4f}")
print(f"Recall:      {res['recall_overall']:.4f}")
print(f"F1 Score:    {res['f1']:.4f}")
print(f"Specificity: {res['spec']:.4f}")
print(f"ROC-AUC:     {res['auc']:.4f}")
print(f"Balanced Accuracy: {res['bal_acc']:.4f}")

print("\n--- NEW: Class-Specific Metrics ---")
print(f"AD Recall:    {res['ad_recall']:.4f} | AD Precision: {res['ad_precision']:.4f}")
print(f"CN Recall:    {res['cn_recall']:.4f} | CN Precision: {res['cn_precision']:.4f}")

print("\nConfusion Matrix:")
print(res['cm'])