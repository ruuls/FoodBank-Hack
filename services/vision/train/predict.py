"""Smoke-test a trained best.pt on one image: prints the count + per-box
class/confidence and saves an annotated copy. Run right after training.

    python predict.py --image /path/to/catch.jpg
    python predict.py --model ../best.pt --image catch.jpg --conf 0.25 --iou 0.6
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../best.pt", help="path to trained weights")
    ap.add_argument("--image", required=True, help="image to run detection on")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6, help="high = keep close/overlapping fish")
    ap.add_argument("--out", default="predict_out.jpg")
    args = ap.parse_args()

    model = YOLO(args.model)
    result = model.predict(args.image, conf=args.conf, iou=args.iou, verbose=False)[0]
    names = result.names

    print(f"detections: {len(result.boxes)}")
    for box in result.boxes:
        cls = names[int(box.cls[0])]
        print(f"  {cls:<12} {float(box.conf[0]):.2f}")

    result.save(filename=args.out)
    print(f"annotated image -> {args.out}")


if __name__ == "__main__":
    main()
