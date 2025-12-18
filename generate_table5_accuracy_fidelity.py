import os
import json
import numpy as np
import pandas as pd

# Base path where JSON files are stored
BASE_PATH = r"F:\Yushun Dong\Project One Uncommited Intern\GNN_OwnVer-main\temp_results\diff\res"

# Configuration
DATASETS = ["Cora", "CiteSeer", "Amazon", "DBLP", "PubMed"]
MODES = ["inductive", "transductive"]
MODE_LABELS = {"inductive": "Inductive", "transductive": "Transductive"}
MODE_PADDING = {"inductive": "  ", "transductive": ""}
SETTINGS = ["test_setting1", "test_setting2", "test_setting3", "test_setting4"]
MODELS = ["gcn", "gat", "graphsage"]

# CSV Output file
OUTPUT_CSV = "surrogate_model_accuracy_fidelity_table5.csv"

def format_mean_std(values):
    if not values:
        return "NA"
    arr = np.array(values)
    return f"{arr.mean():.2f}±{arr.std():.1f}"

def parse_json_file(json_file):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        acc = data.get("original_model_acc", "NA")
        inde_acc = data.get("test_inde_acc_list", [])
        surr_acc = data.get("test_surr_acc_list", [])
        surr_fid = data.get("test_surr_fidelity_list", [])
        return (
            f"{acc:.2f}" if acc != "NA" else "NA",
            format_mean_std(inde_acc),
            format_mean_std(surr_acc),
            format_mean_std(surr_fid),
        )
    except:
        return "NA", "NA", "NA", "NA"

def get_result_row(dataset, mode, model):
    row = [f"{dataset} ({MODE_LABELS[mode]})"]
    target_acc, inde_acc = "NA", "NA"

    for setting_index, setting in enumerate(SETTINGS):
        setting_path = os.path.join(
            BASE_PATH, dataset, mode, "random_mask", "1.0_0.0", "train_setting1", setting
        )
        if not os.path.exists(setting_path):
            acc, inde, sacc, sfid = "NA", "NA", "NA", "NA"
        else:
            # Look for a file starting with the model name
            found_file = None
            for f in os.listdir(setting_path):
                if f.startswith(model) and f.endswith(".json"):
                    found_file = os.path.join(setting_path, f)
                    break
            if found_file:
                acc, inde, sacc, sfid = parse_json_file(found_file)
            else:
                acc, inde, sacc, sfid = "NA", "NA", "NA", "NA"

        if setting_index == 0:
            row.extend([acc, inde])
        row.extend([sacc, sfid])

    return row

def main():
    rows = []
    header = ["Dataset", "Target", "Independent"]
    for i in range(1, 5):
        header.extend([f"Setting {i} Acc", f"Setting {i} Fidelity"])

    for mode in MODES:
        for dataset in DATASETS:
            row = get_result_row(dataset, mode, "gcn")
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Table 5 CSV generated at: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
