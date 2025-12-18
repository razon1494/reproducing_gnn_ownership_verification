import os
import json
import numpy as np
import pandas as pd

# === Configuration ===
DATASETS = ['Cora', 'Citeseer', 'Amazon', 'DBLP', 'Pubmed']
MODELS = ['gcn', 'gat', 'sage']
MODES = ['transductive', 'inductive']
TEST_SETTINGS = [f"test_setting{i}" for i in range(1, 5)]
TRAIN_SETTING = "train_setting1"
ROOT_PATH = os.path.join("temp_results", "diff", "res")

# === Metric Keys ===
metric_keys = {
    "TCA": "original_model_acc",
    "ECA": "mask_model_acc",
    "TBA": "test_surr_acc_list",
    "EBA": "test_surr_fidelity_list"
}

# === Initialize Table Data ===
columns = ["Dataset", "Mode"]
for model in MODELS:
    for bd in ["With Backdoor", "Without Backdoor"]:
        for metric in ["TCA", "ECA", "TBA", "EBA"]:
            columns.append(f"{model.upper()} - {bd} - {metric}")

table_rows = []

# === Process Each Dataset and Mode ===
for dataset in DATASETS:
    for mode in MODES:
        row = [dataset, mode.capitalize()]  # e.g., Transductive / Inductive
        for model in MODELS:
            for is_bd in [True, False]:
                vals = []

                for test_setting in TEST_SETTINGS:
                    test_path = os.path.join(
                        ROOT_PATH, dataset, mode, "random_mask", "1.0_0.0",
                        TRAIN_SETTING, test_setting
                    )
                    if not os.path.exists(test_path):
                        continue

                    for file in os.listdir(test_path):
                        if file.startswith(model):
                            if is_bd and "_0" not in file:
                                continue
                            if not is_bd and "_0" in file:
                                continue

                            with open(os.path.join(test_path, file)) as f:
                                data = json.load(f)
                            vals.append([
                                data.get(metric_keys["TCA"], np.nan),
                                data.get(metric_keys["ECA"], np.nan),
                                np.mean(data.get(metric_keys["TBA"], [np.nan])),
                                np.mean(data.get(metric_keys["EBA"], [np.nan]))
                            ])

                # Compute average if values found
                valid_vals = [v for v in vals if not any(np.isnan(vv) for vv in v)]
                if valid_vals:
                    avg_vals = np.nanmean(valid_vals, axis=0)
                    row += [round(v, 2) for v in avg_vals]
                else:
                    row += ["NA"] * 4

        table_rows.append(row)

# === Export to CSV ===
df = pd.DataFrame(table_rows, columns=columns)
df.to_csv("table3bbox_verification_results.csv", index=False)
print("Saved to table3bbox_verification_results.csv")
