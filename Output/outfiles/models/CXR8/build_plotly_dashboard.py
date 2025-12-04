#!/usr/bin/env python

import json
from pathlib import Path

import pandas as pd
import plotly.express as px

# ============================================================
# 1. CONFIG: FULL PATHS TO YOUR LOG FILES
#    Make sure the last part ("log.txt") matches your actual
#    filename in each experiment directory.
# ============================================================

LOG_FILES = {
    "ResNet-50 Baseline (Exp1)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_stage1_data50p_2-3box_baseline/log.txt",
    "Swin-Tiny Baseline (Exp1)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_swin_tiny_data50p_baseline/log.txt",
    "ViT Baseline (Exp1)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp1_vit_stage1_data50p_2-3box_baseline/log.txt",

    "ResNet-50 2pts Consistency (Exp2)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_stage1_data50p_2-3box_2pts_consLoss100/log.txt",
    "Swin-Tiny 2pts Consistency (Exp2)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_swin_tiny_stage1_data50p_2-3box_2pts_consLoss100/log.txt",
    "ViT 2pts Consistency (Exp2)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp2_vit_stage1_data50p_2-3box_2pts_consLoss100/log.txt",

    "ResNet-50 1pt Unlabel Cons (Exp3)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box/log.txt",
    "Swin-Tiny 1pt Unlabel Cons (Exp3)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_swin_tiny_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box/log.txt",
    "ViT 1pt Unlabel Cons (Exp3)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp3_vit_base_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box/log.txt",

    "ResNet-50 1pt+2pt Cons (Exp4)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp4_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth/log.txt",
    "Swin-Tiny 1pt+2pt Cons (Exp4)": "/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8/exp4_swin_tiny_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth/log.txt",
}

# If your logs are not called "log.txt", change the last part
# for each entry above (e.g. "log_train.txt", "train_log.jsonl", etc.)

# ============================================================
# 2. READ ALL LOGS INTO ONE DATAFRAME
# ============================================================

rows = []

for exp_name, path_str in LOG_FILES.items():
    log_path = Path(path_str)
    if not log_path.exists():
        print(f"[WARN] Missing log file for {exp_name}: {log_path}")
        continue

    print(f"[INFO] Reading {log_path}")
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # keep only pure JSON lines like the ones you pasted
            if not (line.startswith("{") and line.endswith("}")):
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines safely
                continue

            coco = rec.get("test_coco_eval_bbox", [None] * 12)
            ap   = coco[0]
            ap50 = coco[1]
            ap75 = coco[2]

            rows.append({
                "experiment": exp_name,
                "epoch": rec.get("epoch"),
                "train_lr": rec.get("train_lr"),
                "train_loss": rec.get("train_loss"),
                "val_loss": rec.get("test_loss"),
                "train_loss_bbox": rec.get("train_loss_bbox"),
                "train_loss_giou": rec.get("train_loss_giou"),
                "val_loss_bbox": rec.get("test_loss_bbox"),
                "val_loss_giou": rec.get("test_loss_giou"),
                "ap": ap,
                "ap50": ap50,
                "ap75": ap75,
            })

if not rows:
    raise SystemExit("[ERROR] No JSON metrics found. Check file paths / filenames / content.")

df = pd.DataFrame(rows)
df = df.sort_values(["experiment", "epoch"]).reset_index(drop=True)

BASE_DIR = Path("/home/sundeep/Point-Beyond-Class/Output/outfiles/models/CXR8")
csv_out = BASE_DIR / "cxr8_all_experiments_metrics_plotly.csv"
df.to_csv(csv_out, index=False)
print(f"[INFO] Saved combined CSV to {csv_out}")

# ============================================================
# 3. PLOT FUNCTIONS – SAVE AS HTML (NO PNG, NO CHROME)
# ============================================================

PPT_DIR = BASE_DIR / "ppt_dashboard_plots_plotly"
PPT_DIR.mkdir(exist_ok=True)

def make_loss_dashboard(df, out_html):
    """Train vs Val loss for all experiments, vs epoch."""
    df_loss = df.melt(
        id_vars=["experiment", "epoch"],
        value_vars=["train_loss", "val_loss"],
        var_name="loss_type",
        value_name="loss"
    )

    fig = px.line(
        df_loss,
        x="epoch",
        y="loss",
        color="experiment",
        line_dash="loss_type",
        template="plotly_white",
        title="Train / Validation Loss – CXR8, 50% Boxes"
    )
    fig.update_layout(
        legend_title_text="Experiment / Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )

    fig.write_html(str(out_html))
    print(f"[INFO] Saved loss dashboard HTML to: {out_html}")


def make_ap50_dashboard(df, out_html):
    """AP@0.50 vs epoch for all experiments."""
    fig = px.line(
        df,
        x="epoch",
        y="ap50",
        color="experiment",
        markers=True,
        template="plotly_white",
        title="AP@0.50 vs Epoch – CXR8, 50% Boxes"
    )
    fig.update_layout(
        legend_title_text="Experiment",
        xaxis_title="Epoch",
        yaxis_title="AP@0.50 (bbox)"
    )

    fig.write_html(str(out_html))
    print(f"[INFO] Saved AP@0.50 dashboard HTML to: {out_html}")


def make_best_ap50_bar(df, out_html):
    """One bar per experiment: best AP@0.50."""
    best = (
        df.sort_values("ap50", ascending=False)
          .groupby("experiment", as_index=False)
          .first()
    )

    fig = px.bar(
        best,
        x="experiment",
        y="ap50",
        template="plotly_white",
        title="Best AP@0.50 per Experiment – CXR8, 50% Boxes"
    )
    fig.update_layout(
        xaxis_title="Experiment",
        yaxis_title="Best AP@0.50",
        xaxis_tickangle=-25,
    )

    fig.write_html(str(out_html))
    print(f"[INFO] Saved best AP@0.50 bar HTML to: {out_html}")


# ============================================================
# 4. GENERATE ALL THREE DASHBOARDS
# ============================================================

make_loss_dashboard(
    df,
    out_html=PPT_DIR / "cxr8_loss_dashboard.html",
)

make_ap50_dashboard(
    df,
    out_html=PPT_DIR / "cxr8_ap50_dashboard.html",
)

make_best_ap50_bar(
    df,
    out_html=PPT_DIR / "cxr8_best_ap50_bar.html",
)

print(f"[DONE] All HTML dashboards are in: {PPT_DIR}")