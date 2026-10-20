
import os
import sys
import yaml
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from ultralytics import YOLO

TARGET_CLASSES = ["D00", "D10", "D20", "D40"]
ID_TO_CLASS = {idx: name for idx, name in enumerate(TARGET_CLASSES)}
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter_area
    return (inter_area / union) if union > 0 else 0.0

def load_ground_truth(label_file, img_w, img_h):
    gt_list = []
    if not label_file.exists():
        return gt_list

    with open(label_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                xmin = (xc - w / 2.0) * img_w
                ymin = (yc - h / 2.0) * img_h
                xmax = (xc + w / 2.0) * img_w
                ymax = (yc + h / 2.0) * img_h
                gt_list.append({
                    "gt_idx": idx,
                    "class_id": cid,
                    "class_name": ID_TO_CLASS.get(cid, str(cid)),
                    "box": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    "matched": False,
                    "matched_pred_conf": None,
                    "matched_iou": 0.0
                })
    return gt_list

def match_predictions_to_ground_truth(predictions, ground_truths, iou_threshold=0.5):
    sorted_preds = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    matched_preds = []

    for pred in sorted_preds:
        p_class = pred["class_id"]
        p_box = pred["box"]
        best_iou = 0.0
        best_gt_idx = -1

        for g_i, gt in enumerate(ground_truths):
            if gt["class_id"] == p_class:
                iou = compute_box_iou(p_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_i

        is_tp = 0
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            target_gt = ground_truths[best_gt_idx]
            if not target_gt["matched"]:
                is_tp = 1
                target_gt["matched"] = True
                target_gt["matched_pred_conf"] = pred["confidence"]
                target_gt["matched_iou"] = round(best_iou, 4)
            else:
                is_tp = 0

        p_rec = dict(pred)
        p_rec["is_tp"] = is_tp
        p_rec["iou_matched"] = round(best_iou, 4)
        matched_preds.append(p_rec)

    return matched_preds, ground_truths

def main():
    weights_path = Path("/content/runs/detect/baseline_run/weights/best.pt")
    data_yaml_path = Path("/content/yolo_rdd_india/data.yaml")
    results_dir = Path("/content/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 3: EXTRACTING PREDICTIONS (conf=0.001) & MATCHING")
    print("=" * 70)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights_path))

    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    test_img_dir = base_path / data_cfg["test"]
    test_lbl_dir = base_path / "labels" / "test"

    image_paths = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    print(f"Total test images: {len(image_paths)}")

    all_preds = []
    all_gts = []

    for img_path in tqdm(image_paths, desc="Inference on Test Set"):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        gt_boxes = load_ground_truth(test_lbl_dir / f"{img_path.stem}.txt", img_w, img_h)

        results = model.predict(
            source=str(img_path),
            conf=0.001,
            iou=0.7,
            imgsz=640,
            device=device,
            verbose=False
        )[0]

        pred_boxes = []
        for box in results.boxes:
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            pred_boxes.append({
                "image": img_path.name,
                "class_id": cid,
                "class_name": ID_TO_CLASS.get(cid, str(cid)),
                "confidence": round(conf, 6),
                "box": [round(c, 2) for c in xyxy]
            })

        matched_p, updated_g = match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold=0.5)
        all_preds.extend(matched_p)

        for gt in updated_g:
            all_gts.append({
                "image": img_path.name,
                "class_name": gt["class_name"],
                "is_detected": 1 if gt["matched"] else 0,
                "matched_pred_conf": gt["matched_pred_conf"],
                "matched_iou": gt["matched_iou"]
            })

    preds_df = pd.DataFrame(all_preds)
    gt_df = pd.DataFrame(all_gts)

    preds_df.to_csv(results_dir / "test_predictions_matched.csv", index=False)
    gt_df.to_csv(results_dir / "test_ground_truth_matched.csv", index=False)

    print("\n" + "=" * 70)
    print("MATCHING SUMMARY TABLE:")
    print("=" * 70)
    rows = []
    for cname in TARGET_CLASSES:
        cp = preds_df[preds_df["class_name"] == cname] if not preds_df.empty else pd.DataFrame()
        cg = gt_df[gt_df["class_name"] == cname] if not gt_df.empty else pd.DataFrame()
        num_preds = len(cp)
        tp = int(cp["is_tp"].sum()) if not cp.empty else 0
        fp = num_preds - tp
        tot_gt = len(cg)
        missed = tot_gt - int(cg["is_detected"].sum()) if not cg.empty else 0
        rows.append({
            "Class": cname,
            "Total_Preds": num_preds,
            "True_Positives": tp,
            "False_Positives": fp,
            "Total_GroundTruth": tot_gt,
            "Missed_Defects": missed,
            "Recall@0.001": round((tp / tot_gt * 100), 1) if tot_gt > 0 else 0.0
        })

    tot_preds = len(preds_df)
    tot_tp = int(preds_df["is_tp"].sum()) if not preds_df.empty else 0
    tot_gt = len(gt_df)
    rows.append({
        "Class": "OVERALL",
        "Total_Preds": tot_preds,
        "True_Positives": tot_tp,
        "False_Positives": tot_preds - tot_tp,
        "Total_GroundTruth": tot_gt,
        "Missed_Defects": tot_gt - int(gt_df["is_detected"].sum()),
        "Recall@0.001": round((tot_tp / tot_gt * 100), 1) if tot_gt > 0 else 0.0
    })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / "step3_matching_summary.csv", index=False)

    print(f"\nSaved matched table: {results_dir / 'test_predictions_matched.csv'}")
    print("\nSTEP 3 COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
