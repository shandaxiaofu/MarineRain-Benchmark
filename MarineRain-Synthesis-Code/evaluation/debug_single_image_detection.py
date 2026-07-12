#!/usr/bin/env python3
"""Run YOLOv8 detection on a single image and save a visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single-image YOLOv8 diagnostic prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO model path or name.")
    parser.add_argument("--img-size", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--confidence", type=float, default=0.05, help="Confidence threshold.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, 0, or mps.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Output root directory.")
    parser.add_argument("--run-name", default="debug_single_image", help="Output run name.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV failed to read: {image_path}")

    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    print(f"ultralytics={ultralytics.__version__}")
    print(f"image={image_path}, shape={image.shape}")

    model = YOLO(args.model)
    results = model.predict(
        source=str(image_path),
        imgsz=args.img_size,
        conf=args.confidence,
        device=args.device,
        save=True,
        project=str(args.output_dir),
        name=args.run_name,
        exist_ok=True,
        verbose=True,
    )
    result = results[0]
    box_count = 0 if result.boxes is None else int(result.boxes.shape[0])
    print(f"boxes={box_count}")

    if box_count:
        names = result.names if hasattr(result, "names") else model.names
        for xyxy, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = [int(value) for value in xyxy.tolist()]
            if isinstance(names, dict):
                label = names.get(int(class_id), str(class_id))
            else:
                label = names[int(class_id)]
            print(f"{str(label).lower()} {confidence:.3f} [{x1},{y1},{x2},{y2}]")

    print(f"Visualization saved under: {args.output_dir / args.run_name}")


if __name__ == "__main__":
    main()
