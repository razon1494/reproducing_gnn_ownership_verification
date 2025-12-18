import os
import json
import pandas as pd
from collections import defaultdict

# Define the root directory where JSON files are stored
ROOT_DIR = r"F:\Yushun Dong\Project One Uncommited Intern\GNN_OwnVer-main\temp_results\diff\res"

# Where to save the output CSV
OUTPUT_CSV = "table3_summary.csv"

# Target setting to search for JSONs
TARGET_TEST_SETTING = "test_setting1"

# For collecting extracted data
records = []

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    if dirpath.endswith(TARGET_TEST_SETTING):
        dataset = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(dirpath)))))
        )

        for filename in filenames:
            if filename.endswith(".json"):
                filepath = os.path.join(dirpath, filename)

                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    clean_acc = data.get("original_model_acc", None)
                    surr_acc_list = data.get("test_surr_acc_list", None)

                    if clean_acc is None or surr_acc_list is None:
                        print(f"Skipping {filename}: missing metrics")
                        continue

                    # Average surrogate accuracy
                    if isinstance(surr_acc_list[0], list):  # in case of nested list
                        surr_acc = sum([sum(run) for run in surr_acc_list]) / sum([len(run) for run in surr_acc_list])
                    else:
                        surr_acc = sum(surr_acc_list) / len(surr_acc_list)

                    # Parse model and seed from filename: e.g., gcn_224_128_1.json
                    parts = filename.replace(".json", "").split("_")
                    model = parts[0].upper()
                    setting = "Random M"

                    records.append({
                        "Dataset": dataset,
                        "Model": model,
                        "Setting": setting,
                        "Clean Accuracy": clean_acc,
                        "Surrogate Accuracy": surr_acc
                    })

                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

# Create DataFrame
df = pd.DataFrame(records)

# Group and summarize
if not df.empty:
    grouped = df.groupby(["Dataset", "Model", "Setting"]).agg(
        Clean_Acc_Mean=("Clean Accuracy", "mean"),
        Clean_Acc_Std=("Clean Accuracy", "std"),
        Surrogate_Acc_Mean=("Surrogate Accuracy", "mean"),
        Surrogate_Acc_Std=("Surrogate Accuracy", "std")
    ).reset_index()

    # Combine mean ± std
    grouped["Clean Accuracy"] = grouped.apply(
        lambda x: f"{x['Clean_Acc_Mean']:.2f} ± {x['Clean_Acc_Std']:.2f}", axis=1)
    grouped["Surrogate Accuracy"] = grouped.apply(
        lambda x: f"{x['Surrogate_Acc_Mean']:.2f} ± {x['Surrogate_Acc_Std']:.2f}", axis=1)

    # Final output
    final_df = grouped[["Dataset", "Model", "Setting", "Clean Accuracy", "Surrogate Accuracy"]]
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(final_df)
    print(f"\n✅ Summary saved to '{OUTPUT_CSV}'")

else:
    print("❌ No valid data collected.")
