
import os
import sys
import glob
import json
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

TARGET_CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(TARGET_CLASSES)}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def walk_and_inspect_directory(data_path):
    print("\n" + "=" * 70)
    print("1. DIRECTORY INSPECTION")
    print("=" * 70)
    print(f"Inspecting root path: {data_path.resolve()}")

    dir_summary = defaultdict(int)
    extension_counts = Counter()
    sample_annotation_file = None

    for root, dirs, files in os.walk(data_path):
        rel_root = os.path.relpath(root, data_path)
        dir_summary[rel_root] = len(files)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            extension_counts[ext] += 1
            full_path = Path(root) / f
            if ext in [".xml", ".json", ".txt"] and sample_annotation_file is None:
                if "annotation" in str(full_path).lower() or ext == ".xml":
                    sample_annotation_file = full_path

    print("\nDirectory Hierarchy (first 25 subdirectories):")
    for idx, (d, count) in enumerate(sorted(dir_summary.items())):
        if idx < 25:
            print(f"  - {d if d != '.' else '[Root]'}: {count} files")

    print("\nFile counts by extension:")
    for ext, count in extension_counts.most_common():
        ext_label = ext if ext else "[no ext]"
        print(f"  {ext_label:<15}: {count:>6} files")

    if sample_annotation_file and sample_annotation_file.exists():
        print("\n" + "-" * 70)
        print(f"Sample annotation file found: {sample_annotation_file}")
        print("-" * 70)
        with open(sample_annotation_file, "r", encoding="utf-8") as f:
            print(f.read(1200))
            print("\n... [truncated sample]")

def find_dataset_subsets(data_path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
    all_xmls = [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"]
    xml_map = {p.stem: p for p in all_xmls}

    train_candidates = [img for img in all_images if "test" not in str(img).lower()]
    test_candidates = [img for img in all_images if "test" in str(img).lower()]
    if not train_candidates:
        train_candidates = all_images
    return train_candidates, test_candidates, xml_map

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        width = int(float(size.find("width").text)) if size.find("width") is not None else 0
        height = int(float(size.find("height").text)) if size.find("height") is not None else 0
    else:
        width, height = 0, 0

    objects = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        class_name = name_elem.text.strip().upper()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            objects.append({"name": class_name, "bndbox": [xmin, ymin, xmax, ymax]})
        except Exception:
            continue
    return width, height, objects

def analyze_annotations(train_images, test_images, xml_map):
    print("\n" + "=" * 70)
    print("2. ANNOTATION AND DATASET ANALYSIS")
    print("=" * 70)
    print(f"Train images found: {len(train_images)}")
    print(f"Test images found : {len(test_images)}")

    test_with_ann = [img for img in test_images if img.stem in xml_map]
    print(f"Test images with XML annotations: {len(test_with_ann)} / {len(test_images)}")
    if len(test_images) > 0 and len(test_with_ann) == 0:
        print("-> Confirmed: Test images have NO ground-truth annotations (unlabeled test set).")

    total_boxes_per_class = Counter()
    image_records = []
    for img in train_images:
        xml_file = xml_map.get(img.stem)
        if not xml_file:
            continue
        w, h, objs = parse_voc_xml(xml_file)
        if w <= 0 or h <= 0:
            try:
                with Image.open(img) as pil_img:
                    w, h = pil_img.size
            except Exception:
                continue
        target_objs = []
        for obj in objs:
            cname = obj["name"]
            total_boxes_per_class[cname] += 1
            if cname in TARGET_CLASSES:
                target_objs.append(obj)
        image_records.append({
            "image_path": img,
            "width": w,
            "height": h,
            "target_objects": target_objs
        })

    print("\nAll classes found in train annotations:")
    for cname, count in sorted(total_boxes_per_class.items(), key=lambda x: -x[1]):
        status = "KEEP (Target)" if cname in TARGET_CLASSES else "IGNORE"
        print(f"  {cname:<12} : {count:>6} boxes | {status}")

    print("\nTarget Class YOLO Mapping:")
    for cname, cid in CLASS_TO_ID.items():
        print(f"  '{cname}' -> ID {cid} (Total: {total_boxes_per_class[cname]} boxes)")
    return image_records

def split_data_leakage_aware(records, seed=42):
    print("\n" + "=" * 70)
    print("3. LEAKAGE-RESISTANT 70/15/15 SPLITTING")
    print("=" * 70)
    print("Grouping consecutive video frames to prevent spatial/temporal leakage...")
    groups = defaultdict(list)
    for rec in records:
        stem = rec["image_path"].stem
        digits = "".join([c for c in stem if c.isdigit()])
        group_key = f"seq_{int(digits) // 30}" if digits else stem[:7]
        groups[group_key].append(rec)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_g = len(group_keys)
    n_train = int(0.70 * n_g)
    n_val = int(0.15 * n_g)
    train_k = set(group_keys[:n_train])
    val_k = set(group_keys[n_train:n_train + n_val])

    splits = {"train": [], "val": [], "test": []}
    for k, recs in groups.items():
        if k in train_k:
            splits["train"].extend(recs)
        elif k in val_k:
            splits["val"].extend(recs)
        else:
            splits["test"].extend(recs)

    for s_name, s_list in splits.items():
        print(f"  {s_name:<6} split: {len(s_list):>5} images ({len(s_list)/len(records)*100:5.1f}%)")
    return splits

def convert_and_export_yolo(splits, output_dir):
    print("\n" + "=" * 70)
    print("4. EXPORTING YOLO FORMAT DATASET")
    print("=" * 70)
    stats = {s: {"images": len(splits[s]), "boxes": Counter(), "empty": 0} for s in splits}
    for s_name, records in splits.items():
        img_out = output_dir / "images" / s_name
        lbl_out = output_dir / "labels" / s_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for rec in records:
            src_img = rec["image_path"]
            dst_img = img_out / src_img.name
            dst_lbl = lbl_out / f"{src_img.stem}.txt"

            if not dst_img.exists():
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)

            w, h = float(rec["width"]), float(rec["height"])
            lines = []
            for obj in rec["target_objects"]:
                cname = obj["name"]
                if cname not in CLASS_TO_ID:
                    continue
                cid = CLASS_TO_ID[cname]
                xmin, ymin, xmax, ymax = obj["bndbox"]
                xmin, ymin = max(0.0, min(xmin, w)), max(0.0, min(ymin, h))
                xmax, ymax = max(0.0, min(xmax, w)), max(0.0, min(ymax, h))
                bw, bh = xmax - xmin, ymax - ymin
                if bw <= 1.0 or bh <= 1.0:
                    continue
                xc, yc = (xmin + bw / 2.0) / w, (ymin + bh / 2.0) / h
                nw, nh = bw / w, bh / h
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                stats[s_name]["boxes"][cname] += 1

            if not lines:
                stats[s_name]["empty"] += 1
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    print(f"data.yaml written to: {yaml_path}")
    return stats

def print_summary(stats, results_dir):
    print("\n" + "=" * 70)
    print("5. SPLIT AND CLASS COUNTS SUMMARY TABLE")
    print("=" * 70)
    rows = []
    for cname in TARGET_CLASSES:
        row = {"Class": cname, "ID": CLASS_TO_ID[cname]}
        for s in ["train", "val", "test"]:
            row[s] = stats[s]["boxes"][cname]
        row["Total"] = sum([row[s] for s in ["train", "val", "test"]])
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "step1_class_distribution_per_split.csv", index=False)
    print(f"\nSaved CSV table to: {results_dir / 'step1_class_distribution_per_split.csv'}")

if __name__ == "__main__":
    set_seed(42)
    data_path = Path("/content/India")
    if not data_path.exists():
        candidates = list(Path("/content").glob("*India*"))
        if candidates:
            data_path = candidates[0]
    out_dir = Path("/content/yolo_rdd_india")
    res_dir = Path("/content/results")

    walk_and_inspect_directory(data_path)
    train_imgs, test_imgs, xml_map = find_dataset_subsets(data_path)
    records = analyze_annotations(train_imgs, test_imgs, xml_map)
    splits = split_data_leakage_aware(records, seed=42)
    stats = convert_and_export_yolo(splits, out_dir)
    print_summary(stats, res_dir)
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
