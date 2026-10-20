
import os
import sys
import yaml
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import torch
from ultralytics import YOLO

TARGET_CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole"
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    set_seed(42)
    data_yaml = Path("/content/yolo_rdd_india/data.yaml")
    results_dir = Path("/content/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 2: TRAINING YOLOV8N ON T4 GPU")
    print("=" * 70)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({'GPU' if device == 0 else 'CPU'})")

    # 1. Train YOLOv8n
    model = YOLO("yolov8n.pt")
    train_results = model.train(
        data=str(data_yaml),
        epochs=40,
        patience=10,
        imgsz=640,
        batch=16,
        seed=42,
        project="/content/runs/detect",
        name="baseline_run",
        device=device,
        save=True,
        verbose=True,
        val=True
    )

    run_dir = Path("/content/runs/detect/baseline_run")
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "weights" / "last.pt"
    print(f"\nBest checkpoint: {best_weights}")

    # 2. Evaluate strictly on TEST split
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SPLIT ONLY")
    print("=" * 70)
    trained_model = YOLO(str(best_weights))
    test_metrics = trained_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        device=device,
        verbose=True
    )

    box = test_metrics.box
    names = test_metrics.names
    rows = []
    for cid, cname in names.items():
        p = float(box.p[cid]) if cid < len(box.p) else np.nan
        r = float(box.r[cid]) if cid < len(box.r) else np.nan
        ap50 = float(box.ap50[cid]) if cid < len(box.ap50) else np.nan
        ap = float(box.ap[cid]) if cid < len(box.ap) else np.nan
        rows.append({
            "Class": cname,
            "Description": CLASS_DESCRIPTIONS.get(cname, "Defect"),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "AP@0.5": round(ap50, 4),
            "AP@0.5:0.95": round(ap, 4)
        })

    rows.append({
        "Class": "OVERALL",
        "Description": "Mean across all classes",
        "Precision": round(float(box.mp), 4),
        "Recall": round(float(box.mr), 4),
        "AP@0.5": round(float(box.map50), 4),
        "AP@0.5:0.95": round(float(box.map), 4)
    })

    metrics_df = pd.DataFrame(rows)
    print("\nTEST METRICS TABLE:")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(results_dir / "baseline_metrics.csv", index=False)
    print(f"\nSaved metrics to: {results_dir / 'baseline_metrics.csv'}")

    # 3. Copy confusion matrix & training curves
    cm = run_dir / "confusion_matrix_normalized.png"
    if cm.exists():
        shutil.copy2(cm, results_dir / "confusion_matrix.png")
    
    rcsv = run_dir / "results.csv"
    if rcsv.exists():
        df_r = pd.read_csv(rcsv)
        df_r.columns = [c.strip() for c in df_r.columns]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if "train/box_loss" in df_r.columns:
            axes[0].plot(df_r["train/box_loss"], label="Train Box Loss")
        if "val/box_loss" in df_r.columns:
            axes[0].plot(df_r["val/box_loss"], label="Val Box Loss", linestyle="--")
        axes[0].set_title("Box Loss")
        axes[0].grid(True, linestyle=":")
        axes[0].legend()

        map_col = [c for c in df_r.columns if "map50" in c.lower() and "95" not in c.lower()]
        if map_col:
            axes[1].plot(df_r[map_col[0]], label="mAP@0.5", color="green")
        axes[1].set_title("mAP@0.5 over Epochs")
        axes[1].grid(True, linestyle=":")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(results_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"Saved training curves to: {results_dir / 'training_curves.png'}")

    print("\nSTEP 2 COMPLETED!")

if __name__ == "__main__":
    main()
