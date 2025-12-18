import os
import json
import numpy as np
import pandas as pd

# -------- Configuration --------
DATASETS = ["Cora", "Citeseer", "Amazon", "DBLP", "PubMed"]
MODELS = ["gcn", "gat", "sage"]
TEST_SETTINGS = [1, 2, 3, 4]
TRAIN_SETTING = "train_setting1"
BASE_DIR = os.path.join("temp_results", "diff", "res")
OUT_CSV = "table4_fingerprinting_verification_results.csv"

# -------- Utility Functions --------
def extract_metrics(data):
    return data.get("FPR", 0.0), data.get("FNR", 0.0), data.get("Accuracy", 0.0)

def avg_std_str(values):
    if len(values) == 0 or all(v == "NA" for v in values):
        return "NA"
    values = np.array(values, dtype=np.float64)
    return f"{np.mean(values):.2f}±{np.std(values):.1f}"

# -------- Table Generation --------
rows = []

for dataset in DATASETS:
    row_with_A = [f"{dataset} (Cond A)"]
    row_without_A = [f"{dataset} (No A)"]

    for setting in TEST_SETTINGS:
        fprs_A, fnrs_A, accs_A = [], [], []
        fprs_noA, fnrs_noA, accs_noA = [], [], []

        for model in MODELS:
            folder_path = os.path.join(BASE_DIR, dataset, "transductive", "random_mask", "1.0_0.0", TRAIN_SETTING, f"test_setting{setting}")
            if not os.path.exists(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.startswith(model) and file.endswith(".json"):
                    with open(os.path.join(folder_path, file)) as f:
                        data = json.load(f)

                        FPR, FNR, ACC = extract_metrics(data)

                        fprs_A.append(FPR)
                        fnrs_A.append(FNR)
                        accs_A.append(ACC)

                        fprs_noA.append(FPR)
                        fnrs_noA.append(FNR)
                        accs_noA.append(ACC)

        row_with_A.extend([
            avg_std_str(fprs_A),
            avg_std_str(fnrs_A),
            avg_std_str(accs_A),
        ])
        row_without_A.extend([
            avg_std_str(fprs_noA),
            avg_std_str(fnrs_noA),
            avg_std_str(accs_noA),
        ])

    rows.append(row_with_A)
    rows.append(row_without_A)

# -------- Define Proper Column Headers --------
columns = ["Dataset"]
for setting in TEST_SETTINGS:
    columns.extend([
        f"Setting {setting} FPR",
        f"Setting {setting} FNR",
        f"Setting {setting} ACC"
    ])

# -------- Save to CSV with BOM for Excel --------
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"✅ CSV file saved as '{OUT_CSV}' with UTF-8 BOM for Excel compatibility.")
