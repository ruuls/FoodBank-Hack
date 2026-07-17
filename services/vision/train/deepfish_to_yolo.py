"""
Convert the DeepFish (market-tray) COCO dataset into YOLO detection format.

DeepFish is above-water trays, 59 species, annotated as COCO JSON with masks.
This reads the COCO bboxes and writes YOLO detection labels. It DEFAULTS TO
SINGLE-CLASS ('fish') — best count/size for the demo, no risk on rare species.
Pass --multi-class (+ --top-n or --classes) to keep species instead.

Download DeepFish: https://zenodo.org/records/6475675  (unzip; note the COCO
json and the images folder), then:

    python deepfish_to_yolo.py \
        --coco-json /path/to/deepfish/annotations.json \
        --images-dir /path/to/deepfish/images \
        --out ./dataset

    # keep the 8 most-common species instead of single-class:
    #   --multi-class --top-n 8
    # or an explicit allowlist:
    #   --multi-class --classes salmon,tuna,cod,mackerel,sardine

Writes  out/images/{train,val}/…,  out/labels/{train,val}/*.txt,  out/data.yaml
(ready to hand straight to train.py).
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-json", required=True, help="DeepFish COCO annotations JSON")
    ap.add_argument("--images-dir", required=True, help="folder with the tray images")
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--single-class", action="store_true", default=True,
                    help="collapse all species to one 'fish' class (default)")
    ap.add_argument("--multi-class", dest="single_class", action="store_false",
                    help="keep species instead of collapsing to 'fish'")
    ap.add_argument("--top-n", type=int, default=0,
                    help="with --multi-class: keep only the N most common species")
    ap.add_argument("--classes", default="",
                    help="with --multi-class: explicit comma-separated species allowlist")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.coco_json) as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # ---- class set ----
    if args.single_class:
        class_names = ["fish"]
        name_to_idx = None
    else:
        if args.classes:
            keep = [c.strip() for c in args.classes.split(",") if c.strip()]
        elif args.top_n > 0:
            freq = Counter(cats[a["category_id"]] for a in coco["annotations"])
            keep = [name for name, _ in freq.most_common(args.top_n)]
        else:
            keep = sorted(set(cats.values()))
        name_to_idx = {name: i for i, name in enumerate(keep)}
        class_names = keep

    # DeepFish ships images across several zip archives, so they may extract
    # into different subfolders. Build a recursive basename→path index so we
    # find each image regardless of layout instead of silently skipping it.
    img_index: dict[str, str] = {}
    for root, _, fnames in os.walk(args.images_dir):
        for name in fnames:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_index.setdefault(name, os.path.join(root, name))

    # ---- deterministic train/val split ----
    img_ids = sorted(images.keys())
    random.seed(args.seed)
    random.shuffle(img_ids)
    val_ids = set(img_ids[: int(len(img_ids) * args.val_frac)])

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    kept = skipped = 0
    for img_id in img_ids:
        im = images[img_id]
        fn = im["file_name"]
        W, H = im.get("width"), im.get("height")
        src = os.path.join(args.images_dir, fn)
        if not os.path.exists(src):
            src = img_index.get(os.path.basename(fn), "")  # fall back to recursive lookup
        if not src or not os.path.exists(src) or not W or not H:
            skipped += 1
            continue
        split = "val" if img_id in val_ids else "train"

        lines = []
        for a in anns_by_img.get(img_id, []):
            name = cats[a["category_id"]]
            if args.single_class:
                cls = 0
            elif name in name_to_idx:
                cls = name_to_idx[name]
            else:
                continue
            x, y, w, h = a["bbox"]  # COCO bbox = top-left x, y, w, h (absolute px)
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")

        base = os.path.splitext(os.path.basename(fn))[0]
        shutil.copy(src, os.path.join(args.out, "images", split, os.path.basename(fn)))
        with open(os.path.join(args.out, "labels", split, base + ".txt"), "w") as out:
            out.write("\n".join(lines))
        kept += 1

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    with open(os.path.join(args.out, "data.yaml"), "w") as out:
        out.write(
            f"path: {os.path.abspath(args.out)}\n"
            f"train: images/train\nval: images/val\n\nnames:\n{names_block}\n"
        )

    shown = class_names[:8]
    print(f"done: {kept} images ({skipped} missing/invalid skipped), "
          f"{len(class_names)} class(es): {shown}{'…' if len(class_names) > 8 else ''}")
    print(f"data.yaml -> {os.path.abspath(args.out)}/data.yaml")


if __name__ == "__main__":
    main()
