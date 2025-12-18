import os
import json
import numpy as np
import pandas as pd

# Configuration
datasets = ["Cora", "CiteSeer", "Amazon", "DBLP", "PubMed"]
modes = [("inductive", "Induc."), ("transductive", "Trans.")]
root_path = r"F:\Yushun Dong\Project One Uncommited Intern\GNN_OwnVer-main\temp_results\diff\res"
model_tag = "indep"  # identifier for independent models in filenames

# Create structure
columns = ["Dataset", "Training"] + [f"Setting {i} (%)" for i in range(1, 5)]
rows = []

for dataset in datasets:
    for mode, label in modes:
        row = [dataset, label]
        for setting in range(1, 5):
            setting_path = os.path.join(
                root_path, dataset, mode, "random_mask", "1.0_0.0",
                "train_setting1", f"test_setting{setting}"
            )

            fpr_values = []
            if os.path.exists(setting_path):
                for file in os.listdir(setting_path):
                    if model_tag in file and file.endswith(".json"):
                        json_path = os.path.join(setting_path, file)
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            fpr_values.append(data.get("FPR", np.nan))

            if fpr_values:
                mean = np.mean(fpr_values)
                std = np.std(fpr_values)
                row.append(f"{mean:.2f}±{std:.1f}")
            else:
                row.append("0.00±0.0")  # assume 0 if no file found

        rows.append(row)

# Save to CSV
df = pd.DataFrame(rows, columns=columns)
df.to_csv("false_positive_independent_table7.csv", index=False, encoding='utf-8-sig')
print("Saved to false_positive_independent_table7.csv")
