
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

TARGET_CLASSES = ["D00", "D10", "D20", "D40"]

def compute_detection_ece(confidences, is_tp, n_bins=10):
    if len(confidences) == 0:
        return {"ece": np.nan, "bin_confs": np.zeros(n_bins), "bin_prec": np.zeros(n_bins), "bin_counts": np.zeros(n_bins)}

    confidences = np.asarray(confidences, dtype=float)
    is_tp = np.asarray(is_tp, dtype=int)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_confs, bin_prec, bin_counts = [], [], []
    ece = 0.0
    N = len(confidences)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (confidences >= low) & (confidences <= high) if i == n_bins - 1 else (confidences >= low) & (confidences < high)
        cnt = int(np.sum(mask))
        bin_counts.append(cnt)
        if cnt > 0:
            m_conf = float(np.mean(confidences[mask]))
            m_prec = float(np.mean(is_tp[mask]))
            bin_confs.append(m_conf)
            bin_prec.append(m_prec)
            ece += (cnt / N) * abs(m_conf - m_prec)
        else:
            bin_confs.append((low + high) / 2.0)
            bin_prec.append(0.0)

    return {"ece": float(ece), "bin_confs": np.array(bin_confs), "bin_prec": np.array(bin_prec), "bin_counts": np.array(bin_counts), "bin_edges": bin_edges}

def fit_temperature_scaling(confidences, is_tp, eps=1e-6):
    p_clamped = np.clip(confidences, eps, 1.0 - eps)
    logits = logit(p_clamped)

    def nll(t):
        p_s = expit(logits / t)
        p_s = np.clip(p_s, eps, 1.0 - eps)
        return -np.mean(is_tp * np.log(p_s) + (1 - is_tp) * np.log(1 - p_s))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    opt_t = float(res.x)

    def calibrator(c):
        return expit(logit(np.clip(c, eps, 1.0 - eps)) / opt_t)

    return opt_t, calibrator

def fit_isotonic(confidences, is_tp):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(confidences, is_tp)
    return iso, lambda c: iso.predict(c)

def main():
    results_dir = Path("/content/results")
    preds_csv = results_dir / "test_predictions_matched.csv"
    df = pd.read_csv(preds_csv)

    print("=" * 70)
    print("STEP 4: CONFIDENCE CALIBRATION ANALYSIS (ECE)")
    print("=" * 70)

    # 1. Full uncalibrated ECE
    full_ece = compute_detection_ece(df["confidence"].values, df["is_tp"].values)
    print(f"Overall Uncalibrated Test ECE: {full_ece['ece']:.4f}")

    # Plot full reliability diagram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, 1], [0, 1], "k--", label="Ideal (y=x)")
    mask = full_ece["bin_counts"] > 0
    ax1.plot(full_ece["bin_confs"][mask], full_ece["bin_prec"][mask], "o-", color="red", label=f"Raw YOLOv8 (ECE={full_ece['ece']:.4f})")
    ax1.set_ylabel("Observed Precision")
    ax1.set_title("Uncalibrated Reliability Diagram (Full Test Set)")
    ax1.grid(True, linestyle=":")
    ax1.legend()

    centers = (full_ece["bin_edges"][:-1] + full_ece["bin_edges"][1:]) / 2.0
    ax2.bar(centers, full_ece["bin_counts"], width=0.08, color="gray", alpha=0.7)
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(results_dir / "reliability_diagram_uncalibrated.png", dpi=150)
    plt.close()

    # 2. Split test set by image into 50% Calibration / 50% Evaluation
    unique_imgs = sorted(df["image"].unique())
    np.random.seed(42)
    np.random.shuffle(unique_imgs)
    n_cal = len(unique_imgs) // 2
    cal_imgs = set(unique_imgs[:n_cal])
    eval_imgs = set(unique_imgs[n_cal:])

    cal_df = df[df["image"].isin(cal_imgs)].copy()
    eval_df = df[df["image"].isin(eval_imgs)].copy()
    print(f"\nPartitioned by Image (No Leakage):")
    print(f"  - Calibration Half: {len(cal_imgs)} images, {len(cal_df)} predictions")
    print(f"  - Evaluation Half : {len(eval_imgs)} images, {len(eval_df)} predictions")

    # 3. Fit on Calibration Half ONLY
    opt_t, temp_cal = fit_temperature_scaling(cal_df["confidence"].values, cal_df["is_tp"].values)
    iso_model, iso_cal = fit_isotonic(cal_df["confidence"].values, cal_df["is_tp"].values)
    print(f"\nFitted Temperature T: {opt_t:.4f}")

    # 4. Evaluate on Evaluation Half ONLY
    eval_raw = eval_df["confidence"].values
    eval_tp = eval_df["is_tp"].values
    eval_temp = temp_cal(eval_raw)
    eval_iso = iso_cal(eval_raw)

    eval_df["conf_temp_scaled"] = eval_temp
    eval_df["conf_isotonic"] = eval_iso
    eval_df.to_csv(results_dir / "test_predictions_calibrated_eval_half.csv", index=False)

    ece_raw = compute_detection_ece(eval_raw, eval_tp)["ece"]
    ece_temp = compute_detection_ece(eval_temp, eval_tp)["ece"]
    ece_iso = compute_detection_ece(eval_iso, eval_tp)["ece"]

    comp_rows = [{
        "Class": "OVERALL",
        "Box_Count": len(eval_df),
        "Raw_ECE": round(ece_raw, 4),
        "Temp_Scaled_ECE": round(ece_temp, 4),
        "Isotonic_ECE": round(ece_iso, 4),
        "Best_Reduction": round(ece_raw - min(ece_temp, ece_iso), 4)
    }]

    for cname in TARGET_CLASSES:
        c_sub = eval_df[eval_df["class_name"] == cname]
        if len(c_sub) > 0:
            cr = compute_detection_ece(c_sub["confidence"].values, c_sub["is_tp"].values)["ece"]
            ct = compute_detection_ece(c_sub["conf_temp_scaled"].values, c_sub["is_tp"].values)["ece"]
            ci = compute_detection_ece(c_sub["conf_isotonic"].values, c_sub["is_tp"].values)["ece"]
            comp_rows.append({
                "Class": cname,
                "Box_Count": len(c_sub),
                "Raw_ECE": round(cr, 4),
                "Temp_Scaled_ECE": round(ct, 4),
                "Isotonic_ECE": round(ci, 4),
                "Best_Reduction": round(cr - min(ct, ci), 4)
            })

    comp_df = pd.DataFrame(comp_rows)
    print("\n" + "=" * 70)
    print("BEFORE VS AFTER ECE COMPARISON (OUT-OF-SAMPLE EVALUATION HALF):")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(results_dir / "calibration_ece_comparison.csv", index=False)

    # Plot Comparison Diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal (y=x)")
    c_raw = compute_detection_ece(eval_raw, eval_tp)
    c_temp = compute_detection_ece(eval_temp, eval_tp)
    c_iso = compute_detection_ece(eval_iso, eval_tp)
    ax.plot(c_raw["bin_confs"][c_raw["bin_counts"]>0], c_raw["bin_prec"][c_raw["bin_counts"]>0], "s-", color="red", label=f"Raw YOLOv8 (ECE={ece_raw:.4f})")
    ax.plot(c_temp["bin_confs"][c_temp["bin_counts"]>0], c_temp["bin_prec"][c_temp["bin_counts"]>0], "o-", color="blue", label=f"Temp Scaling (ECE={ece_temp:.4f})")
    ax.plot(c_iso["bin_confs"][c_iso["bin_counts"]>0], c_iso["bin_prec"][c_iso["bin_counts"]>0], "^-", color="green", label=f"Isotonic (ECE={ece_iso:.4f})")
    ax.set_title("Calibration Comparison on Evaluation Half")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Observed Precision")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "reliability_diagram_comparison.png", dpi=150)
    plt.close()

    print(f"\nSaved calibration comparison table to: {results_dir / 'calibration_ece_comparison.csv'}")
    print("STEP 4 COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
