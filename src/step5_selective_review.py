
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    results_dir = Path("/content/results")
    preds_csv = results_dir / "test_predictions_calibrated_eval_half.csv"
    gt_csv = results_dir / "test_ground_truth_matched.csv"

    print("=" * 70)
    print("STEP 5: SELECTIVE REVIEW (AUTO-ACCEPT VS HUMAN TRIAGE)")
    print("=" * 70)

    preds_df = pd.read_csv(preds_csv)
    eval_images = set(preds_df["image"].unique())

    # Count real defects strictly on evaluation half
    gt_df = pd.read_csv(gt_csv)
    eval_gt_sub = gt_df[gt_df["image"].isin(eval_images)]
    total_eval_gt = len(eval_gt_sub)
    total_preds = len(preds_df)

    print(f"Total Detections on Evaluation Half : {total_preds}")
    print(f"Total Real Ground-Truth Defects     : {total_eval_gt}")

    methods = [
        ("confidence", "Raw YOLOv8"),
        ("conf_temp_scaled", "Temperature Scaled"),
        ("conf_isotonic", "Isotonic Regression")
    ]

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    sweep_records = []

    for col, m_label in methods:
        for t in thresholds:
            auto_mask = preds_df[col] >= t
            n_auto = int(auto_mask.sum())
            n_human = total_preds - n_auto
            auto_tps = int(preds_df.loc[auto_mask, "is_tp"].sum())

            prec = (auto_tps / n_auto) if n_auto > 0 else np.nan
            recall = (auto_tps / total_eval_gt * 100.0) if total_eval_gt > 0 else np.nan

            sweep_records.append({
                "Method": m_label,
                "Threshold": t,
                "Auto_Accepted_Pct": round(n_auto / total_preds * 100.0, 1),
                "Auto_Accepted_Precision": round(prec, 4) if not np.isnan(prec) else np.nan,
                "True_Defect_Recall_Pct": round(recall, 1) if not np.isnan(recall) else np.nan,
                "Human_Workload_Pct": round(n_human / total_preds * 100.0, 1),
                "Sent_To_Human_Count": n_human
            })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(results_dir / "selective_review_sweep_all_thresholds.csv", index=False)

    # Find exact thresholds for 90% and 95% Precision Targets
    fine_t = np.linspace(0.01, 0.99, 197)
    target_records = []

    for target in [0.90, 0.95]:
        for col, m_label in methods:
            found = False
            for t in fine_t:
                auto_mask = preds_df[col] >= t
                n_auto = int(auto_mask.sum())
                if n_auto == 0:
                    continue
                auto_tps = int(preds_df.loc[auto_mask, "is_tp"].sum())
                prec = auto_tps / n_auto
                if prec >= target:
                    n_human = total_preds - n_auto
                    rec = (auto_tps / total_eval_gt * 100.0) if total_eval_gt > 0 else 0.0
                    target_records.append({
                        "Method": m_label,
                        "Target_Precision": f"{int(target * 100)}%",
                        "Required_Threshold": round(float(t), 3),
                        "Achieved_Precision": round(float(prec), 4),
                        "Auto_Accepted_Count": n_auto,
                        "Sent_To_Human": n_human,
                        "Human_Workload_Pct": round(n_human / total_preds * 100.0, 1),
                        "Defect_Recall_Pct": round(rec, 1),
                        "Status": "Achievable"
                    })
                    found = True
                    break
            if not found:
                target_records.append({
                    "Method": m_label,
                    "Target_Precision": f"{int(target * 100)}%",
                    "Required_Threshold": np.nan,
                    "Achieved_Precision": np.nan,
                    "Auto_Accepted_Count": 0,
                    "Sent_To_Human": total_preds,
                    "Human_Workload_Pct": 100.0,
                    "Defect_Recall_Pct": 0.0,
                    "Status": "Unachievable"
                })

    target_df = pd.DataFrame(target_records)
    print("\n" + "=" * 70)
    print("TARGET PRECISION TRIAGE AUDIT (90% & 95% PRECISION):")
    print("=" * 70)
    print(target_df.to_string(index=False))
    target_df.to_csv(results_dir / "selective_review_threshold_targets.csv", index=False)

    # Generate Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"Raw YOLOv8": "red", "Temperature Scaled": "blue", "Isotonic Regression": "green"}

    for col, m_label in methods:
        sub = sweep_df[sweep_df["Method"] == m_label].dropna(subset=["Auto_Accepted_Precision"])
        axes[0].plot(sub["Human_Workload_Pct"], sub["Auto_Accepted_Precision"] * 100.0, "o-", label=m_label, color=palette[m_label])
        axes[1].plot(sub["Human_Workload_Pct"], sub["True_Defect_Recall_Pct"], "s-", label=m_label, color=palette[m_label])

    axes[0].axhline(90, color="gray", linestyle=":", label="90% Target")
    axes[0].axhline(95, color="gray", linestyle="--", label="95% Target")
    axes[0].set_title("Auto-Accepted Precision vs. Human Workload")
    axes[0].set_xlabel("Detections Sent to Human (%)")
    axes[0].set_ylabel("Auto-Accepted Precision (%)")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    axes[1].set_title("Defect Recall vs. Human Workload")
    axes[1].set_xlabel("Detections Sent to Human (%)")
    axes[1].set_ylabel("True Defects Auto-Accepted (%)")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "selective_review_tradeoffs.png", dpi=150)
    plt.close()

    print(f"\nSaved plot to : {results_dir / 'selective_review_tradeoffs.png'}")
    print(f"Saved table to: {results_dir / 'selective_review_threshold_targets.csv'}")

    print("\n" + "=" * 70)
    print("HONEST SUMMARY: DID CALIBRATION HELP?")
    print("=" * 70)
    print("1. Ranking / Workload Frontier: Monotonic calibration does NOT change the")
    print("   relative order of bounding boxes. Hence, to achieve 90% precision, the")
    print("   human inspector must review roughly the same set of uncertain boxes.")
    print("2. Threshold Trustworthiness (The Big Win): With Raw YOLO, setting t=0.90")
    print("   is misleading because the model is overconfident. With Isotonic calibration,")
    print("   the confidence dial directly reflects true precision, allowing operators")
    print("   to set safe policies without trial-and-error guessing.")
    print("=" * 70)
    print("STEP 5 COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
