"""End-to-end implementation of the MarineRain synthesis equation.

The paper-authoritative formation model implemented here is:

    O = T * B + L * (1 - T) + T * R_gc
    T = exp(-beta * d)
    R_gc = R_orig ** gamma

All image-valued terms are represented in the normalized [0, 1] range.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from utils.depth_decoder import DepthDecoder
from utils.resnet_encoder import ResnetEncoder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FORMATION_EQUATION = "O=T*B+L*(1-T)+T*R_gc; T=exp(-beta*d)"


@dataclass(frozen=True)
class PipelineConfig:
    clean_dir: Path
    rain_dir: Path
    output_dir: Path
    model_dir: Path
    gamma: float = 3.5
    beta_min: float = 1.0
    beta_max: float = 1.2
    airlight_min: float = 0.8
    airlight_max: float = 1.0
    samples_per_image: int = 1
    seed: int = 1234
    device: str = "auto"
    recursive: bool = True
    save_depth: bool = True
    save_gamma_rain: bool = False
    output_format: str = "png"
    jpeg_quality: int = 95
    overwrite: bool = False


def list_images(directory: Path, recursive: bool) -> List[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but it is not available")
    return torch.device(requested)


def set_reproducibility(seed: int) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return rng


def read_rgb_float(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def write_rgb_float(path: Path, image_rgb: np.ndarray, jpeg_quality: int = 95) -> None:
    image_u8 = np.clip(np.rint(image_rgb * 255.0), 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    params: Sequence[int] = ()
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality)
    if not cv2.imwrite(str(path), image_bgr, list(params)):
        raise IOError(f"Failed to write image: {path}")


def write_relative_depth(path: Path, relative_depth: np.ndarray) -> None:
    depth_u16 = np.clip(np.rint(relative_depth * 65535.0), 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), depth_u16):
        raise IOError(f"Failed to write relative-depth image: {path}")


class Monodepth2Estimator:
    """Load Monodepth2 once and return per-image normalized relative depth."""

    def __init__(self, model_dir: Path, device_name: str):
        self.model_dir = model_dir
        self.device = select_device(device_name)

        encoder_path = model_dir / "encoder.pth"
        decoder_path = model_dir / "depth.pth"
        missing = [path for path in (encoder_path, decoder_path) if not path.is_file()]
        if missing:
            expected = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing required Monodepth2 weight file(s): {expected}")

        self.encoder = ResnetEncoder(18, False)
        loaded_encoder = torch.load(str(encoder_path), map_location=self.device)
        self.feed_height = int(loaded_encoder["height"])
        self.feed_width = int(loaded_encoder["width"])
        encoder_state = {
            key: value
            for key, value in loaded_encoder.items()
            if key in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(encoder_state)
        self.encoder.to(self.device).eval()

        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(torch.load(str(decoder_path), map_location=self.device))
        self.decoder.to(self.device).eval()

    def estimate_relative_depth(self, clean_rgb: np.ndarray) -> np.ndarray:
        height, width = clean_rgb.shape[:2]
        resized = cv2.resize(
            clean_rgb,
            (self.feed_width, self.feed_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        tensor = transforms.ToTensor()(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.encoder(tensor)
            output = self.decoder(features)
            disparity = output[("disp", 0)]
            disparity = torch.nn.functional.interpolate(
                disparity,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        disparity_np = disparity.squeeze().detach().cpu().numpy().astype(np.float32)
        minimum = float(disparity_np.min())
        maximum = float(disparity_np.max())
        if maximum - minimum <= np.finfo(np.float32).eps:
            normalized_disparity = np.zeros_like(disparity_np, dtype=np.float32)
        else:
            normalized_disparity = (disparity_np - minimum) / (maximum - minimum)

        # Monodepth2 predicts disparity (near regions are generally brighter).
        # The manuscript uses the inverted, normalized map as relative scene depth d.
        return np.clip(1.0 - normalized_disparity, 0.0, 1.0).astype(np.float32)


def crop_rain_to_background(
    rain_rgb: np.ndarray,
    target_height: int,
    target_width: int,
    rng: random.Random,
) -> np.ndarray:
    """Resize to cover the target, then take a reproducible random crop."""

    rain_height, rain_width = rain_rgb.shape[:2]
    scale = max(target_height / rain_height, target_width / rain_width, 1.0)
    if scale > 1.0:
        resized_width = max(target_width, int(round(rain_width * scale)))
        resized_height = max(target_height, int(round(rain_height * scale)))
        rain_rgb = cv2.resize(
            rain_rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rain_height, rain_width = rain_rgb.shape[:2]

    top = rng.randint(0, rain_height - target_height)
    left = rng.randint(0, rain_width - target_width)
    return rain_rgb[top : top + target_height, left : left + target_width]


def gamma_correct_rain(rain_rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Apply R_gc = (R_orig ** gamma) in normalized [0, 1] space."""

    return np.power(np.clip(rain_rgb, 0.0, 1.0), gamma).astype(np.float32)


def synthesize_with_atmospheric_model(
    clean_rgb: np.ndarray,
    gamma_rain_rgb: np.ndarray,
    relative_depth: np.ndarray,
    beta: float,
    airlight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the paper's unified depth-aware rain--fog formation model.

    O = T*B + L*(1-T) + T*R_gc, where T = exp(-beta*d).
    """

    if clean_rgb.shape != gamma_rain_rgb.shape:
        raise ValueError("Clean image and rain layer must have identical HxWx3 shapes")
    if relative_depth.shape != clean_rgb.shape[:2]:
        raise ValueError("Relative depth must match the clean image height and width")

    transmission = np.exp(-beta * relative_depth).astype(np.float32)
    transmission_3c = transmission[:, :, None]
    observed = (
        transmission_3c * clean_rgb
        + airlight * (1.0 - transmission_3c)
        + transmission_3c * gamma_rain_rgb
    )
    return np.clip(observed, 0.0, 1.0).astype(np.float32), transmission


def prepare_output_directories(config: PipelineConfig) -> dict:
    metadata_path = config.output_dir / "metadata.csv"
    if metadata_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output metadata already exists: {metadata_path}. "
            "Use a new directory or pass --overwrite."
        )

    directories = {
        "clean": config.output_dir / "clean",
        "rainy": config.output_dir / "rainy",
    }
    if config.save_depth:
        directories["depth"] = config.output_dir / "relative_depth"
    if config.save_gamma_rain:
        directories["gamma_rain"] = config.output_dir / "gamma_rain"

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def run_pipeline(config: PipelineConfig) -> None:
    clean_images = list_images(config.clean_dir, config.recursive)
    rain_images = list_images(config.rain_dir, config.recursive)
    if not clean_images:
        raise RuntimeError(f"No supported clean images found in {config.clean_dir}")
    if not rain_images:
        raise RuntimeError(f"No supported rain-layer images found in {config.rain_dir}")

    directories = prepare_output_directories(config)
    rng = set_reproducibility(config.seed)
    estimator = Monodepth2Estimator(config.model_dir, config.device)
    output_suffix = ".png" if config.output_format == "png" else ".jpg"
    metadata_path = config.output_dir / "metadata.csv"

    fieldnames = [
        "pair_id",
        "clean_output",
        "rainy_output",
        "relative_depth_output",
        "gamma_rain_output",
        "clean_source",
        "rain_source",
        "gamma",
        "beta",
        "airlight",
        "seed",
        "model_dir",
        "relative_depth_definition",
        "formation_equation",
    ]

    total = len(clean_images) * config.samples_per_image
    completed = 0
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total, desc="MarineRain synthesis", unit="pair") as progress:
            for clean_index, clean_path in enumerate(clean_images, start=1):
                clean_rgb = read_rgb_float(clean_path)
                relative_depth = estimator.estimate_relative_depth(clean_rgb)
                height, width = clean_rgb.shape[:2]

                for sample_index in range(1, config.samples_per_image + 1):
                    pair_id = f"{clean_index:06d}_{clean_path.stem}_s{sample_index:02d}"
                    rain_path = rng.choice(rain_images)
                    rain_rgb = read_rgb_float(rain_path)
                    rain_patch = crop_rain_to_background(rain_rgb, height, width, rng)
                    gamma_rain = gamma_correct_rain(rain_patch, config.gamma)
                    beta = rng.uniform(config.beta_min, config.beta_max)
                    airlight = rng.uniform(config.airlight_min, config.airlight_max)
                    rainy_rgb, _ = synthesize_with_atmospheric_model(
                        clean_rgb,
                        gamma_rain,
                        relative_depth,
                        beta,
                        airlight,
                    )

                    clean_output = directories["clean"] / f"{pair_id}{output_suffix}"
                    rainy_output = directories["rainy"] / f"{pair_id}{output_suffix}"
                    write_rgb_float(clean_output, clean_rgb, config.jpeg_quality)
                    write_rgb_float(rainy_output, rainy_rgb, config.jpeg_quality)

                    depth_output = ""
                    if config.save_depth:
                        depth_path = directories["depth"] / f"{pair_id}.png"
                        write_relative_depth(depth_path, relative_depth)
                        depth_output = str(depth_path.relative_to(config.output_dir))

                    gamma_rain_output = ""
                    if config.save_gamma_rain:
                        gamma_rain_path = directories["gamma_rain"] / f"{pair_id}{output_suffix}"
                        write_rgb_float(gamma_rain_path, gamma_rain, config.jpeg_quality)
                        gamma_rain_output = str(
                            gamma_rain_path.relative_to(config.output_dir)
                        )

                    writer.writerow(
                        {
                            "pair_id": pair_id,
                            "clean_output": str(clean_output.relative_to(config.output_dir)),
                            "rainy_output": str(rainy_output.relative_to(config.output_dir)),
                            "relative_depth_output": depth_output,
                            "gamma_rain_output": gamma_rain_output,
                            "clean_source": str(clean_path.relative_to(config.clean_dir)),
                            "rain_source": str(rain_path.relative_to(config.rain_dir)),
                            "gamma": f"{config.gamma:.8g}",
                            "beta": f"{beta:.8g}",
                            "airlight": f"{airlight:.8g}",
                            "seed": config.seed,
                            "model_dir": str(config.model_dir),
                            "relative_depth_definition": "1 - per-image normalized disparity",
                            "formation_equation": FORMATION_EQUATION,
                        }
                    )
                    completed += 1
                    progress.update(1)

    print(f"Completed {completed} paired samples.")
    print(f"Outputs: {config.output_dir}")
    print(f"Metadata: {metadata_path}")
