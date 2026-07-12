# MarineRain Synthesis Pipeline

Official synthesis code for the **MarineRain Benchmark**, a depth-aware maritime rain--fog image generation pipeline built from clean scene images and physically captured rain-streak layers.

The pipeline combines monocular relative-depth estimation, Gamma correction, atmospheric scattering, and depth-dependent rain attenuation to generate paired clean/rainy images for maritime image restoration research.

## Overview

Given a clean background image `B`, a captured rain-streak layer `R`, and a relative scene-depth map `d`, the pipeline generates a rainy observation `O` using:

```text
R_gc = R ^ gamma
T    = exp(-beta * d)
O    = T * B + L * (1 - T) + T * R_gc
```

where:

- `R_gc` is the Gamma-corrected rain layer;
- `T` is the depth-dependent transmission map;
- `beta` is the atmospheric scattering coefficient;
- `L` is the normalized atmospheric-light value;
- `d` is the inverted and normalized Monodepth2 disparity map.

The default synthesis settings are:

| Parameter | Default setting |
|---|---:|
| Gamma exponent `gamma` | `3.5` |
| Scattering coefficient `beta` | Uniformly sampled from `[1.0, 1.2]` |
| Atmospheric light `L` | Uniformly sampled from `[0.8, 1.0]` |
| Random seed | `1234` |

## Pipeline

```text
Clean background
      │
      ├── Monodepth2 inference
      │        └── normalized disparity → inverted relative depth d
      │
Captured rain layer
      │
      ├── resize and random crop
      └── Gamma correction with gamma = 3.5
               │
               ▼
Depth-aware atmospheric formation model
               │
               ▼
Paired clean image, rainy image, relative depth, and metadata
```

## Repository Structure

```text
MarineRain-Synthesis-Code/
├── generate_marinerain.py       # Command-line entry point
├── marinerain_pipeline.py       # Depth estimation and synthesis implementation
├── environments.txt             # Reference Python environment
├── .gitignore
│
├── utils/                       # Monodepth2 network modules
│   ├── depth_decoder.py
│   ├── layers.py
│   └── resnet_encoder.py
│
├── models/                      # Local model-weight directory
│   └── README.md
│
├── evaluation/                  # YOLOv8-based ship-detection evaluation
└── tools/                       # Image-region visualization utilities
```

The recommended synthesis entry point is `generate_marinerain.py`.

## Requirements

Python 3.10 is recommended.

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r environments.txt
```

Core dependencies include PyTorch, torchvision, OpenCV, NumPy, Pillow, and tqdm. The environment file also includes the packages used by the optional evaluation utilities.

## Monodepth2 Weights

Place a pretrained Monodepth2 encoder and depth decoder in a local model directory:

```text
models/
└── mono+stereo_1024x320/
    ├── encoder.pth
    └── depth.pth
```

The model directory can also be specified explicitly with `--model-dir`.

Model weights are not tracked by Git. The synthesis pipeline requires only `encoder.pth` and `depth.pth`; pose-estimation weights are not used.

## Data Preparation

Organize clean images and captured rain layers in separate directories:

```text
data/
├── clean/
│   ├── sea_0001.jpg
│   └── sea_0002.jpg
└── rain_layers/
    ├── rain_0001.png
    └── rain_0002.png
```

Supported image formats are:

```text
.jpg  .jpeg  .png  .bmp  .tif  .tiff  .webp
```

Subdirectories are searched recursively by default.

## Usage

Run the complete synthesis pipeline from the project directory:

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --model-dir models/mono+stereo_1024x320
```

For each clean image, the program:

1. estimates Monodepth2 disparity;
2. normalizes and inverts the disparity to obtain relative depth;
3. randomly selects and crops a captured rain layer;
4. applies Gamma correction;
5. samples `beta` and atmospheric light `L`;
6. applies the unified depth-aware rain--fog formation model;
7. writes the paired images, relative-depth map, and synthesis metadata.

### Multiple Samples per Background

Generate multiple independently randomized rainy observations for each clean image:

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --samples-per-image 2
```

### Select an Inference Device

The default `auto` mode prefers CUDA, then Apple MPS, and finally CPU.

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --device cuda
```

Supported values are:

```text
auto  cpu  cuda  mps
```

### Save Corrected Rain Patches

Use `--save-gamma-rain` to save the cropped and Gamma-corrected rain layer associated with each generated pair:

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --save-gamma-rain
```

### List All Options

```bash
python generate_marinerain.py --help
```

Important options include:

```text
--gamma
--beta-min / --beta-max
--airlight-min / --airlight-max
--samples-per-image
--seed
--device
--recursive / --no-recursive
--save-depth / --no-save-depth
--save-gamma-rain / --no-save-gamma-rain
--output-format png|jpg
--jpeg-quality
--overwrite
```

Keep the default Gamma, scattering, and atmospheric-light parameters to reproduce the synthesis settings reported for MarineRain.

## Output Format

The default output structure is:

```text
outputs/marinerain/
├── clean/
│   └── 000001_sea_0001_s01.png
├── rainy/
│   └── 000001_sea_0001_s01.png
├── relative_depth/
│   └── 000001_sea_0001_s01.png
└── metadata.csv
```

If `--save-gamma-rain` is enabled, the output additionally contains:

```text
gamma_rain/
```

Clean and rainy images use identical filenames to provide directly aligned training pairs.

Relative-depth maps are stored as 16-bit PNG files. They represent:

```text
d = 1 - per-image normalized disparity
```

They are relative scene-depth maps rather than metric depth estimates.

## Reproducibility Metadata

Each row of `metadata.csv` records:

- the generated pair identifier;
- clean and rainy output paths;
- clean-image and rain-layer source paths;
- the Gamma exponent;
- the sampled `beta` value;
- the sampled atmospheric-light value;
- the random seed;
- the Monodepth2 model directory;
- the relative-depth definition;
- the image-formation equation.

Using the same ordered inputs, model weights, dependency environment, and random seed reproduces the rain-layer selection, crop position, `beta`, and atmospheric-light sampling sequence.

## Evaluation Utilities

The `evaluation/` directory contains optional YOLOv8-based tools for boat detection and VOC-style AP evaluation. These scripts are separate from the synthesis pipeline and require the evaluation dependencies listed in `environments.txt`.

Run boat detection on a directory of images:

```bash
python evaluation/ship_detection_and_voc_ap.py \
  --image-dir outputs/marinerain/rainy \
  --out-csv outputs/detection/boats.csv \
  --draw-dir outputs/detection/visualizations \
  --model yolov8s.pt \
  --device cpu
```

To additionally compute VOC-style boat AP, provide the annotation directory:

```bash
python evaluation/ship_detection_and_voc_ap.py \
  --image-dir outputs/marinerain/rainy \
  --out-csv outputs/detection/boats.csv \
  --voc-ann-dir data/VOC/Annotations \
  --iou-thresh-ap 0.5
```

Debug detection on a single image:

```bash
python evaluation/debug_single_image_detection.py \
  --image data/example.jpg \
  --model yolov8s.pt \
  --device cpu
```

## Visualization Tool

`tools/crop_marked_regions.py` detects red rectangles in a reference image, applies the same regions to a directory of aligned images, and exports both crops and boxed previews:

```bash
python tools/crop_marked_regions.py \
  --reference-image figures/reference_with_red_boxes.png \
  --input-dir figures/comparison_images \
  --output-dir outputs/region_crops \
  --pattern "*.png"
```

## Notes

- Model weights, datasets, and generated outputs are excluded from version control.
- The Monodepth2 output is used as relative depth after per-image disparity normalization and inversion.
- `metadata.csv` prevents silent parameter loss when generating randomized datasets.
- An existing output directory containing `metadata.csv` is protected from accidental reuse unless `--overwrite` is specified.

## Citation


