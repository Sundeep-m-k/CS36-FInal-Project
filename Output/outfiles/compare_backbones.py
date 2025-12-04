#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Global heading for titles
HEADING = "VindrCXR-50p based on Point-DETR"

# Friendly names for groups
GROUP_TITLES = {
    "EXP1": "EXP1: Baseline",
    "EXP2": "EXP2: Baseline + Multipoint",
    "EXP3": "EXP3: Baseline + Symmetric Consistency",
    "EXP4": "EXP4: Baseline + MP + SC",
}

# ---------------------------------------------------
# Helper: map internal key -> backbone display label
# ---------------------------------------------------
def backbone_label(model_key: str) -> str:
    """
    model_key examples: 'EXP1_ResNet', 'EXP2_Swin', 'EXP3_ViT'
    """
    suffix = model_key.split("_", 1)[1] if "_" in model_key else model_key
    suffix_lower = suffix.lower()
    if "resnet" in suffix_lower:
        return "ResNet-50"
    if "swin" in suffix_lower:
        return "Swin-Tiny"
    if "vit" in suffix_lower:
        return "ViT-Base"
    return suffix


# -----------------------------
# Helper: Safe load of log.txt
# -----------------------------
def load_log_safe(path, model_name):
    if os.path.isdir(path):
        path = os.path.join(path, "log.txt")

    if not os.path.exists(path):
        print(f"[WARN] Missing log: {model_name} -> {path}")
        return None

    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue

            # Expand COCO eval bbox metrics
            if "test_coco_eval_bbox" in entry and isinstance(
                entry["test_coco_eval_bbox"], list
            ):
                COCO = [
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
                vals = entry["test_coco_eval_bbox"]
                for i, name in enumerate(COCO):
                    if i < len(vals):
                        entry[f"test_{name}"] = vals[i]

            entry["model_key"] = model_name    # internal key
            entry["backbone"] = backbone_label(model_name)
            records.append(entry)

    if len(records) == 0:
        print(f"[WARN] No valid JSON in {model_name}")
        return None

    df = pd.DataFrame(records)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="ignore")
    return df


# -------------------------------
# Paths for all backbone variants
# -------------------------------
LOGS = {
    # EXP1
    "EXP1_ResNet": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_stage1_data50p_2-3box_baseline",
    "EXP1_Swin": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_swin_tiny_data50p_baseline",
    "EXP1_ViT": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_vit_stage1_data50p_2-3box_baseline",

    # EXP2
    "EXP2_ResNet": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_stage1_data50p_2-3box_2pts_consLoss100",
    "EXP2_Swin": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_swin_tiny_stage1_data50p_2-3box_2pts_consLoss100",
    "EXP2_ViT": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_vit_stage1_data50p_2-3box_2pts_consLoss100",

    # EXP3
    "EXP3_ResNet": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box",
    "EXP3_Swin": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_swin_tiny_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box",
    "EXP3_ViT": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_vit_base_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box",

    # EXP4
    "EXP4_ResNet": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp4_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth",
    "EXP4_Swin": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp4_swin_tiny_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth",
    # Vit exp4 may not exist yet; script will skip if missing:
    "EXP4_ViT": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp4_vit_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth",
}

# -----------------------------
# Group experiments
# -----------------------------
GROUPS = {
    "EXP1": ["EXP1_ResNet", "EXP1_Swin", "EXP1_ViT"],
    "EXP2": ["EXP2_ResNet", "EXP2_Swin", "EXP2_ViT"],
    "EXP3": ["EXP3_ResNet", "EXP3_Swin", "EXP3_ViT"],
    "EXP4": ["EXP4_ResNet", "EXP4_Swin", "EXP4_ViT"],
}

# Metrics to plot
METRICS = ["train_loss", "test_loss", "test_AP", "test_AP50", "test_AR100"]

os.makedirs("plots_backbone_compare", exist_ok=True)


# -----------------------------
# Plotting function per group
# -----------------------------
def plot_group(group_name, group_dfs):
    group_title = GROUP_TITLES.get(group_name, group_name)
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        plotted = False

        for model_key, df in group_dfs.items():
            if metric not in df.columns:
                continue
            d = df.sort_values("epoch")
            label = d["backbone"].iloc[0]  # ResNet-50 / Swin-Tiny / ViT-Base
            plt.plot(d["epoch"], d[metric], marker="o", label=label)
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{HEADING}\n{group_title} — Backbone Comparison ({metric})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        out = f"plots_backbone_compare/{group_name}_{metric}.png"
        plt.savefig(out, dpi=250)
        plt.close()
        print(f"[Saved plot] {out}")


# -----------------------------
# AP50 table per group (PNG+CSV)
# -----------------------------
def ap50_table_group(group_name, group_dfs):
    # Collect final-epoch AP50 per backbone
    rows = []
    for model_key, df in group_dfs.items():
        if "test_AP50" not in df.columns:
            continue
        d = df.sort_values("epoch")
        last = d.tail(1).iloc[0]
        backbone = last["backbone"]
        epoch = int(last["epoch"])
        ap50 = float(last["test_AP50"])
        rows.append((backbone, epoch, ap50))

    if not rows:
        print(f"[INFO] No AP50 data for {group_name}, skipping table.")
        return

    # Build DataFrame and sort ascending by AP50 (%)
    table_df = pd.DataFrame(rows, columns=["Model", "Epoch", "AP50"])
    table_df["AP50_percent"] = table_df["AP50"] * 100.0
    table_df = table_df.sort_values("AP50_percent", ascending=True)

    # Save CSV
    csv_path = f"plots_backbone_compare/{group_name}_AP50_table.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"[Saved CSV table] {csv_path}")

    # Create PNG table
    models = table_df["Model"].tolist()
    cell_text = []
    for _, row in table_df.iterrows():
        cell_text.append(
            [
                row["Model"],
                f"{int(row['Epoch'])}",
                f"{row['AP50']:.4f}",
                f"{row['AP50_percent']:.2f}%",
            ]
        )

    # Find best (highest AP50)
    best_idx = table_df["AP50_percent"].idxmax()
    best_model = table_df.loc[best_idx, "Model"]
    best_row_index = models.index(best_model)

    fig, ax = plt.subplots(figsize=(8, 1.3 + 0.5 * len(models)))
    ax.axis("off")

    col_labels = ["Model", "Epoch", "AP50", "AP50 (%)"]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 1.25)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if r == 0:  # header
            cell.set_fontsize(12)
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    # Highlight best row (add 1 for header row)
    highlight_row = best_row_index + 1
    for c in range(len(col_labels)):
        cell = table[highlight_row, c]
        cell.set_facecolor("#c6f5c6")
        cell.set_text_props(weight="bold")

    group_title = GROUP_TITLES.get(group_name, group_name)
    ax.set_title(
        f"{HEADING}\n{group_title} — Final AP50 (%) per Backbone",
        pad=12,
        fontsize=14,
        fontweight="bold",
    )

    png_path = f"plots_backbone_compare/{group_name}_AP50_table.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved PNG table] {png_path}")


# -----------------------------
# Main: Load, Group, Plot
# -----------------------------
def main():
    loaded = {}
    # Load all logs
    for name, path in LOGS.items():
        df = load_log_safe(path, name)
        if df is not None:
            loaded[name] = df

    # For each experiment group: make plots + AP50 table
    for group_name, model_list in GROUPS.items():
        group_dfs = {m: loaded[m] for m in model_list if m in loaded}
        if len(group_dfs) == 0:
            print(f"[SKIP] No logs for {group_name}")
            continue

        print(f"\n--- {group_name}: plotting & tables ---")
        plot_group(group_name, group_dfs)
        ap50_table_group(group_name, group_dfs)


if __name__ == "__main__":
    main()
