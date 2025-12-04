#!/usr/bin/env python3
"""
parse_logs_and_plot.py

Reads multiple DETR-style log.txt files (one JSON per line), extracts metrics,
expands COCO eval metrics, and generates:

- Combined CSV with all epochs & models: all_metrics.csv
- Final-epoch comparison table: final_epoch_metrics.csv
- Best-AP comparison table: best_AP_metrics.csv
- Final AP50 summary with percentage values: final_AP50_percent.csv
- Line plots per metric comparing all models: plots/*.png
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 0. GLOBAL HEADING
# ---------------------------------------------------------------------
HEADING = "VindrCXR-50p based on Point-DETR"

# ---------------------------------------------------------------------
# 1. CONFIG: your 4 models + log paths
#    (keys are the NAMES that will appear in tables/plots)
# ---------------------------------------------------------------------
LOG_FILES = {
    "Baseline": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_stage1_data50p_2-3box_baseline/log.txt",
    "Baseline+Multipoint": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_stage1_data50p_2-3box_2pts_consLoss100",
    "Baseline+Symmetric consistency": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box",
    "Baseline+MP+SC": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp4_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth",
}

# If some paths are directories, automatically append 'log.txt'
for k, p in list(LOG_FILES.items()):
    if os.path.isdir(p):
        LOG_FILES[k] = os.path.join(p, "log.txt")

# COCO metric names in the order stored in test_coco_eval_bbox
COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


# ---------------------------------------------------------------------
# 2. Utility: load one log into a DataFrame
# ---------------------------------------------------------------------
def load_log_to_df(log_path: str, model_name: str) -> pd.DataFrame:
    """
    Read a log.txt file with one JSON object per line, expand
    test_coco_eval_bbox into separate columns, and return a DataFrame.
    """
    records = []

    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found for model {model_name}: {log_path}")

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip non-JSON lines if any
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Expand COCO metrics if present
            if "test_coco_eval_bbox" in entry and entry["test_coco_eval_bbox"] is not None:
                coco_vals = entry["test_coco_eval_bbox"]
                if isinstance(coco_vals, list):
                    for i, name in enumerate(COCO_METRIC_NAMES):
                        if i < len(coco_vals):
                            entry[f"test_{name}"] = coco_vals[i]
                        else:
                            entry[f"test_{name}"] = None

            entry["model"] = model_name
            records.append(entry)

    if not records:
        raise ValueError(f"No valid JSON lines parsed from {log_path}")

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------
# 3. Main: load all logs, merge to single DataFrame
# ---------------------------------------------------------------------
def main():
    all_dfs = []
    for model_name, log_path in LOG_FILES.items():
        print(f"Loading {model_name} from {log_path}")
        df_model = load_log_to_df(log_path, model_name)
        all_dfs.append(df_model)

    df = pd.concat(all_dfs, ignore_index=True)

    # Ensure epoch is numeric and sorted
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values(["model", "epoch"])

    # Save full metrics
    df.to_csv("all_metrics.csv", index=False)
    print("Saved all_metrics.csv")

    # -----------------------------------------------------------------
    # 4. Tables: final-epoch metrics & best-AP metrics
    # -----------------------------------------------------------------

    # Final epoch per model
    final_rows = (
        df.sort_values("epoch")
        .groupby("model", as_index=False)
        .tail(1)
        .set_index("model")
    )

    # Add AP50 percentage for final epoch
        # Final AP50 as a dedicated table (percent) + nice PNG table
        # Final AP50 as a dedicated table (percent) + nice PNG table
    if "test_AP50" in final_rows.columns:
        final_ap50 = final_rows[["epoch", "test_AP50"]].copy()
        final_ap50["test_AP50_percent"] = final_ap50["test_AP50"] * 100.0

        # Sort ascending by AP50%
        final_ap50_sorted = final_ap50.sort_values("test_AP50_percent", ascending=True)

        # Save CSV
        final_ap50_sorted.to_csv("final_AP50_percent.csv")
        print("\n=== Final AP50 (percent) per model (ascending) ===")
        print(final_ap50_sorted)
        print("Saved final_AP50_percent.csv")

        # ---------------------------------------------------------
        # Create visually nice PNG table (with model column)
        # ---------------------------------------------------------
        import matplotlib.pyplot as plt

        # Data for table
        models = final_ap50_sorted.index.tolist()
        col_labels = ["Model", "Epoch", "AP50", "AP50 (%)"]
        cell_text = []
        for model, row in final_ap50_sorted.iterrows():
            cell_text.append([
                model,
                f"{int(row['epoch'])}",
                f"{row['test_AP50']:.4f}",
                f"{row['test_AP50_percent']:.2f}%",
            ])

        # Best (highest) AP50%
        best_model = final_ap50_sorted["test_AP50_percent"].idxmax()
        best_row_idx = models.index(best_model)

        n_rows = len(models)

        fig, ax = plt.subplots(figsize=(8, 1.3 + 0.5 * n_rows))
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )

        # Basic styling
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.25)

        # Style header + grid + highlight best row
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0.8)

            # header row
            if r == 0:
                cell.set_fontsize(12)
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")

        # highlight row (add 1 because header is row 0)
        highlight_row = best_row_idx + 1
        for c in range(len(col_labels)):
            cell = table[highlight_row, c]
            cell.set_facecolor("#c6f5c6")
            cell.set_text_props(weight="bold")

        ax.set_title(
            f"{HEADING}\nFinal AP50 (%) per Model (ascending, best highlighted)",
            pad=12,
            fontsize=14,
            fontweight="bold",
        )

        import os
        os.makedirs("plots", exist_ok=True)
        table_path = os.path.join("plots", "final_AP50_table.png")

        # bbox_inches="tight" prevents cutting
        plt.savefig(table_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved PNG table: {table_path}")




    # -----------------------------------------------------------------
    # 5. Plots: line charts comparing models for each metric
    # -----------------------------------------------------------------
    metrics_to_plot = [
        "train_loss",
        "test_loss",
        "train_loss_bbox",
        "test_loss_bbox",
        "train_loss_giou",
        "test_loss_giou",
        "test_AP",
        "test_AP50",
        "test_AP75",
        "test_AR1",
        "test_AR10",
        "test_AR100",
    ]

    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    os.makedirs("plots", exist_ok=True)

    for metric in metrics_to_plot:
        plt.figure()
        for model_name, d in df.groupby("model"):
            d = d.sort_values("epoch")
            plt.plot(d["epoch"], d[metric], marker="o", label=model_name)

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{HEADING}\n{metric} vs Epoch")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        out_path = os.path.join("plots", f"{metric}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
