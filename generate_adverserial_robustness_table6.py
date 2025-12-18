import os
import json
import numpy as np
import pandas as pd

# Settings
datasets = ["Cora", "CiteSeer", "Amazon", "DBLP", "PubMed"]
modes = ["inductive", "transductive"]
model = "gcn"  # can be changed to 'gat' or 'graphsage' if needed
root_path = r"F:\Yushun Dong\Project One Uncommited Intern\GNN_OwnVer-main\temp_results\diff\res"

# Create column structure
columns = ["Dataset", "Ori. ACC (%)"]
for setting in range(1, 5):
    columns.extend([
        f"Setting {setting} FPR", 
        f"Setting {setting} FNR", 
        f"Setting {setting} ACC"
    ])

# Collect results
rows = []
for mode in modes:
    for dataset in datasets:
        row = [f"{dataset} ({'Inductive' if mode == 'inductive' else 'Transductive'})"]
        ori_acc_collected = False

        for setting in range(1, 5):
            setting_path = os.path.join(
                root_path, dataset, mode, "random_mask", "1.0_0.0",
                "train_setting1", f"test_setting{setting}"
            )
            acc_values, fpr_values, fnr_values = [], [], []

            for file in os.listdir(setting_path):
                if file.startswith(model) and file.endswith(".json"):
                    with open(os.path.join(setting_path, file), "r") as f:
                        data = json.load(f)
                        acc_values.append(data.get("Accuracy", np.nan))
                        fpr_values.append(data.get("FPR", np.nan))
                        fnr_values.append(data.get("FNR", np.nan))
                        if not ori_acc_collected:
                            row.append(round(data.get("original_model_acc", np.nan), 2))
                            ori_acc_collected = True

            # Compute mean ± std for each metric
            for values in [fpr_values, fnr_values, acc_values]:
                values = np.array(values)
                if len(values) > 0:
                    mean = np.mean(values)
                    std = np.std(values)
                    row.append(f"{mean:.2f}±{std:.2f}")
                else:
                    row.append("NA")

        rows.append(row)

# Save to CSV
df = pd.DataFrame(rows, columns=columns)
df.to_csv("adversarial_robustness_table6.csv", index=False, encoding='utf-8-sig')
print("Saved to adversarial_robustness_table6.csv")
