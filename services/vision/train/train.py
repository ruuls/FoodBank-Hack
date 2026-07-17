"""
Train YOLOv12 for TideLift catch detection.

Detects fish in an ABOVE-WATER handling scene (tote / conveyor / deck) and
returns boxes → the intake layer maps those to count + size (species comes from
the log, or from a multi-class model). See ../README about the service, and
../../.. for how best.pt plugs in.

Usage (on your GPU box):
    pip install ultralytics
    python train.py --data data.yaml --epochs 100 --imgsz 640 --batch 16
    # then copy runs/detect/tidelift-catch/weights/best.pt -> ../best.pt

Notes:
  * Dataset MUST be above-water with bounding boxes and multiple fish per image.
    Do NOT train on underwater footage — wrong visual domain for dockside sorting.
  * If `yolov12s.pt` isn't available in your ultralytics version, pass
    --model yolo11s.pt (YOLO11 is rock-solid and identical to drive).
  * Single-class ('fish') is the default and pairs with the log's species /
    a separate species step. A multi-class dataset works too — the mapper reads
    the dominant detected class.
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml", help="YOLO dataset yaml")
    ap.add_argument("--model", default="yolov12s.pt", help="pretrained weights (or yolo11s.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="tidelift-catch")
    ap.add_argument("--device", default="0", help="GPU id ('0'), '0,1' for multi-GPU, or 'cpu'")
    ap.add_argument("--cache", action="store_true", help="cache images in RAM for faster epochs")
    ap.add_argument("--export-onnx", action="store_true", help="also export best.pt -> ONNX for edge")
    args = ap.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,   # -1 = Ultralytics AutoBatch (recommended for yolov12x)
        device=args.device,
        cache=args.cache,
        name=args.name,
        patience=25,
        # Robustness to real dockside conditions — wet decks, glare, low/rainy
        # light — so the model generalizes past clean dataset photos:
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,   # value jitter = lighting variation
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        mosaic=1.0,  # helps dense/overlapping scenes
        mixup=0.1,
    )

    # Validation mAP — quote this to judges as the accuracy number.
    metrics = model.val(data=args.data)
    print(f"\nmAP@50    = {metrics.box.map50:.3f}")
    print(f"mAP@50-95 = {metrics.box.map:.3f}")
    print("best weights -> runs/detect/%s/weights/best.pt" % args.name)

    if args.export_onnx:
        model.export(format="onnx")  # for a phone/edge deployment later


if __name__ == "__main__":
    main()
