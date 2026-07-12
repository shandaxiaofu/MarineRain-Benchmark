#!/usr/bin/env python3
"""Canonical command-line entry point for MarineRain synthesis."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "mono+stereo_1024x320"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired MarineRain images with Monodepth2 relative depth, "
            "Gamma-corrected physical rain layers, and the paper's unified "
            "atmospheric image-formation equation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--clean-dir",
        type=Path,
        required=True,
        help="Directory containing clean background images.",
    )
    parser.add_argument(
        "--rain-dir",
        type=Path,
        required=True,
        help="Directory containing captured rain-layer images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for paired clean/rainy outputs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Monodepth2 directory containing encoder.pth and depth.pth.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=3.5,
        help="Rain-layer Gamma exponent reported in the paper.",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=1.0,
        help="Minimum atmospheric scattering coefficient.",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=1.2,
        help="Maximum atmospheric scattering coefficient.",
    )
    parser.add_argument(
        "--airlight-min",
        type=float,
        default=0.8,
        help="Minimum normalized atmospheric-light value.",
    )
    parser.add_argument(
        "--airlight-max",
        type=float,
        default=1.0,
        help="Maximum normalized atmospheric-light value.",
    )

    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=1,
        help="Number of independently randomized rainy pairs per clean image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed controlling rain selection, crop positions, beta, and airlight.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch inference device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search clean and rain directories recursively.",
    )
    parser.add_argument(
        "--save-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the normalized relative-depth map used by the formula.",
    )
    parser.add_argument(
        "--save-gamma-rain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save each cropped, Gamma-corrected rain patch for inspection.",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "jpg"),
        default="png",
        help="Image format for clean/rainy pairs.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when --output-format=jpg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an output directory that already has metadata.csv.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.gamma <= 0:
        parser.error("--gamma must be greater than zero")
    if args.beta_min < 0 or args.beta_max < args.beta_min:
        parser.error("beta range must satisfy 0 <= beta-min <= beta-max")
    if not (0 <= args.airlight_min <= args.airlight_max <= 1):
        parser.error("airlight range must satisfy 0 <= min <= max <= 1")
    if args.samples_per_image < 1:
        parser.error("--samples-per-image must be at least 1")
    if not (1 <= args.jpeg_quality <= 100):
        parser.error("--jpeg-quality must be between 1 and 100")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    try:
        from marinerain_pipeline import PipelineConfig, run_pipeline
    except ModuleNotFoundError as exc:
        dependency = exc.name or "a required package"
        parser.error(
            f"Missing dependency '{dependency}'. Install the reference environment with "
            "`python -m pip install -r environments.txt`."
        )

    config = PipelineConfig(
        clean_dir=args.clean_dir.expanduser().resolve(),
        rain_dir=args.rain_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        model_dir=args.model_dir.expanduser().resolve(),
        gamma=args.gamma,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        airlight_min=args.airlight_min,
        airlight_max=args.airlight_max,
        samples_per_image=args.samples_per_image,
        seed=args.seed,
        device=args.device,
        recursive=args.recursive,
        save_depth=args.save_depth,
        save_gamma_rain=args.save_gamma_rain,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
