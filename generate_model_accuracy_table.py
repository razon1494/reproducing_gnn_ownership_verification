import os
import json
import pandas as pd
import numpy as np

# Settings
DATASETS = ['Cora', 'Citeseer', 'Amazon', 'DBLP', 'PubMed']
ROOT_PATH = r"F:\Yushun Dong\Project One Uncommited Intern\GNN_OwnVer-main\temp_results\diff\res"
MASK_TYPE = "random_mask"
MASK_RATIO = "1.0_0.0"
TRAIN_SETTING = "train_setting1"
TEST_SETTINGS = [f"test_setting{i}" for i in range(1, 5)]
MODE = "transductive"

# Function to extract accuracy and fidelity
def extract_metrics(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        acc = np.round(np.mean(data["acc_test"]), 2)
        acc_std = np.round(np.std(data["acc_test"]), 1)
        fid = np.round(np.mean(data["fidelity_test"]), 2)
        fid_std = np.round(np.std(data["fidelity_test"]), 1)
        return f"{acc}±{acc_std}", f"{fid}±{fid_std}"
    except:
        return "NA", "NA"

# Function to construct file name
def get_file_name(model, surrogate_type, setting):
    suffix = {
        "target": "224_128_0",
        "independent": "288_128_0",
        "shadow": {
            1: "224_128_0",
            2: "224_128_1",
            3: "224_128_2",
            4: "288_128_0"
        }
    }
    return f"{model}_{suffix[surrogate_type] if surrogate_type != 'shadow' else suffix[surrogate_type][setting]}.json"

# Main processing
rows = []
for dataset in DATASETS:
    base_path = os.path.join(ROOT_PATH, dataset, MODE, MASK_TYPE, MASK_RATIO, TRAIN_SETTING)

    row_inductive = [f"{dataset} (Inductive)"]
    row_transductive = [f"{dataset} (Transductive)"]

    for model in ["gcn", "gat", "sage"]:
        # Target
        target_path = os.path.join(base_path, "test_setting1", get_file_name(model, "target", 1))
        acc, _ = extract_metrics(target_path)
        row_inductive.append(acc)
        row_transductive.append(acc)

        # Independent
        indep_path = os.path.join(base_path, "test_setting1", get_file_name(model, "independent", 1))
        acc, _ = extract_metrics(indep_path)
        row_inductive.append(acc)
        row_transductive.append(acc)

        # Shadow settings
        for setting in range(1, 5):
            shadow_path = os.path.join(base_path, f"test_setting{setting}", get_file_name(model, "shadow", setting))
            acc, fid = extract_metrics(shadow_path)
            row_inductive.append(acc)
            row_inductive.append(fid)
            row_transductive.append(acc)
            row_transductive.append(fid)

    rows.append(row_inductive)
    rows.append(row_transductive)

# Column headers
columns = ["Dataset"]
for model in ["GCN", "GAT", "GraphSAGE"]:
    columns.extend([
        f"{model} Target (%)",
        f"{model} Independent (%)",
        f"{model} Setting I Accuracy",
        f"{model} Setting I Fidelity",
        f"{model} Setting II Accuracy",
        f"{model} Setting II Fidelity",
        f"{model} Setting III Accuracy",
        f"{model} Setting III Fidelity",
        f"{model} Setting IV Accuracy",
        f"{model} Setting IV Fidelity"
    ])

# Save to CSV
df = pd.DataFrame(rows, columns=columns)
csv_name = "surrogate_model_accuracy_fidelity_table5.csv"
df.to_csv(csv_name, index=False)
print(f"CSV saved to {csv_name}")
