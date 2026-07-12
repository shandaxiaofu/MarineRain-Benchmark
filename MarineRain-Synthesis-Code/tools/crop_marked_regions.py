#!/usr/bin/env python3
"""Extract marked comparison regions using red boxes from a reference image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect red boxes in a reference image and apply the same crop regions "
            "to aligned comparison images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        required=True,
        help="Image containing the red reference boxes.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing aligned images to crop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for crops and boxed previews.",
    )
    parser.add_argument("--pattern", default="*.png", help="Input filename glob pattern.")
    parser.add_argument("--min-box-width", type=int, default=10, help="Minimum detected box width.")
    parser.add_argument("--min-box-height", type=int, default=10, help="Minimum detected box height.")
    parser.add_argument("--red-min", type=int, default=150, help="Minimum red-channel value.")
    parser.add_argument("--green-max", type=int, default=80, help="Maximum green-channel value.")
    parser.add_argument("--blue-max", type=int, default=80, help="Maximum blue-channel value.")
    return parser


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    return image


def detect_red_boxes(
    reference: np.ndarray,
    red_min: int,
    green_max: int,
    blue_max: int,
    min_width: int,
    min_height: int,
) -> List[Box]:
    lower_red = np.array([0, 0, red_min], dtype=np.uint8)
    upper_red = np.array([blue_max, green_max, 255], dtype=np.uint8)
    mask = cv2.inRange(reference, lower_red, upper_red)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > min_width and height > min_height:
            boxes.append((x, y, x + width, y + height))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def main() -> None:
    args = build_parser().parse_args()
    reference_path = args.reference_image.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {reference_path}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    reference = read_image(reference_path)
    reference_height, reference_width = reference.shape[:2]
    boxes = detect_red_boxes(
        reference,
        red_min=args.red_min,
        green_max=args.green_max,
        blue_max=args.blue_max,
        min_width=args.min_box_width,
        min_height=args.min_box_height,
    )
    if not boxes:
        raise RuntimeError("No red reference boxes were detected")

    image_paths = sorted(path for path in input_dir.glob(args.pattern) if path.is_file())
    if not image_paths:
        raise RuntimeError(f"No input images matched pattern '{args.pattern}' in {input_dir}")

    crops_dir = output_dir / "crops"
    boxed_dir = output_dir / "boxed"
    crops_dir.mkdir(parents=True, exist_ok=True)
    boxed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected {len(boxes)} reference box(es): {boxes}")
    for image_path in image_paths:
        image = read_image(image_path)
        if image.shape[:2] != (reference_height, reference_width):
            raise ValueError(
                f"Image size mismatch for {image_path}: expected "
                f"{reference_width}x{reference_height}, got "
                f"{image.shape[1]}x{image.shape[0]}"
            )

        boxed = image.copy()
        for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 0, 255), 2)
            crop = image[y1:y2, x1:x2]
            crop_path = crops_dir / f"{image_path.stem}_crop{index}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise IOError(f"Failed to write crop: {crop_path}")

        boxed_path = boxed_dir / f"{image_path.stem}_boxed.png"
        if not cv2.imwrite(str(boxed_path), boxed):
            raise IOError(f"Failed to write boxed preview: {boxed_path}")

    print(f"Processed {len(image_paths)} image(s).")
    print(f"Crops: {crops_dir}")
    print(f"Boxed previews: {boxed_dir}")


if __name__ == "__main__":
    main()
