import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# ========= User Config =========
# Clean images (used for depth estimation)
IMG_DIR = Path(r"D:\ocean\input")

# Pre-existing haze images (rain will be applied on these)
HAZE_DIR = Path(r"D:\ocean\haze_m_o")

# Where to write (or read) depth maps (8-bit png)
DEPTH_DIR = Path(r"D:\ocean\mask_m_s")

# Folder of rain-streak texture images
RAIN_TEXTURE_DIR = Path(r"D:\Monadepth\test_gamma\drop")

# Output directory for final rain-only composites
OUT_DIR = Path(r"D:\Monadepth\test_gamma\RF")

# Monodepth2 model folder (leave as-is if your weights live here)
MODEL_DIR = Path(r"models/mono+stereo_1024x320")

# If you already have depth maps in DEPTH_DIR, set this True to skip depth estimation
SKIP_DEPTH_ESTIMATION = True

# Depth→rain mapping:
#   "far_more_rain":   deeper pixels (smaller disparity) get stronger rain (default; matches your old behavior)
#   "near_more_rain":  nearer pixels get stronger rain
DEPTH_RAIN_MODE = "near_more_rain"

# Rain blending strength ranges
ALPHA_RANGE = (1.5, 2.0)   # scales rain texture intensity
BETA_RANGE  = (0.9, 1.1)   # controls depth falloff in e^{-beta * depth}
RNG_SEED = 1234            # make results reproducible; set None for full randomness
# ===============================


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def _ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def _read_rgb(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _write_bgr(path: Path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


# ---------------- Monodepth2 (disparity→normalized 8-bit map) ----------------
def run_depth_estimation():
    from utils.resnet_encoder import ResnetEncoder
    from utils.depth_decoder import DepthDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_path = MODEL_DIR / "encoder.pth"
    depth_decoder_path = MODEL_DIR / "depth.pth"

    if not encoder_path.exists() or not depth_decoder_path.exists():
        raise FileNotFoundError(
            f"Missing monodepth2 weights in {MODEL_DIR}. "
            f"Expected encoder.pth and depth.pth"
        )

    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(str(encoder_path), map_location=device)

    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device).eval()

    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(str(depth_decoder_path), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device).eval()

    images = _list_images(IMG_DIR)
    _ensure_dir(DEPTH_DIR)

    with tqdm(total=len(images), desc="(1/2) Estimating depth") as pbar:
        for img_path in images:
            rgb = _read_rgb(img_path)
            H, W = rgb.shape[:2]

            resized = cv2.resize(rgb, (feed_width, feed_height), interpolation=cv2.INTER_LANCZOS4)
            tensor = transforms.ToTensor()(resized).unsqueeze(0).to(device)

            with torch.no_grad():
                feats = encoder(tensor)
                outputs = depth_decoder(feats)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(disp, (H, W), mode="bilinear", align_corners=False)

            disp_np = disp_resized.squeeze().cpu().numpy().astype(np.float32)
            disp_norm = cv2.normalize(disp_np, None, 0, 1, cv2.NORM_MINMAX)
            depth_u8 = (disp_norm * 255.0).astype(np.uint8)

            out_name = (DEPTH_DIR / img_path.name).with_suffix(".png")
            cv2.imwrite(str(out_name), depth_u8)

            pbar.update(1)


# ---------------- Rain texture helpers ----------------
def load_rain_texture_any(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Rain texture not found: {path}")
    tex = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if tex is None:
        raise FileNotFoundError(f"Failed to read rain texture: {path}")
    return tex.astype(np.float32) / 255.0


def crop_or_tile_texture(tex, target_h, target_w):
    h, w = tex.shape[:2]

    scale = max(target_h / h, target_w / w)
    if scale > 1.0:
        new_w = max(int(w * scale * 1.1), target_w)
        new_h = max(int(h * scale * 1.1), target_h)
        tex = cv2.resize(tex, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = tex.shape[:2]

    if h < target_h or w < target_w:
        rep_y = target_h // h + 1
        rep_x = target_w // w + 1
        tex = np.tile(tex, (rep_y, rep_x, 1))
        h, w = tex.shape[:2]

    y = random.randint(0, h - target_h)
    x = random.randint(0, w - target_w)
    return tex[y:y + target_h, x:x + target_w]


def add_depth_aware_rain(rgb, depth01, rain_bgr):
    H, W = rgb.shape[:2]
    rain_bgr = cv2.resize(rain_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    rain = cv2.cvtColor((rain_bgr * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = rgb.astype(np.float32) / 255.0

    alpha = random.uniform(*ALPHA_RANGE)
    beta = random.uniform(*BETA_RANGE)

    D = depth01.astype(np.float32)
    if DEPTH_RAIN_MODE == "near_more_rain":
        D = 1.0 - D

    k = np.exp(-beta * D)
    k3 = k[:, :, None]

    comp = img + alpha * rain * k3
    comp = np.clip(comp, 0.0, 1.0)
    return (comp * 255.0).astype(np.uint8)


def main():
    _set_seed(RNG_SEED)

    clean_images = _list_images(IMG_DIR)
    haze_images = _list_images(HAZE_DIR)
    rain_files = _list_images(RAIN_TEXTURE_DIR)

    if len(clean_images) == 0:
        raise RuntimeError(f"No clean images in {IMG_DIR}")
    if len(haze_images) == 0:
        raise RuntimeError(f"No haze images in {HAZE_DIR}")
    if len(rain_files) == 0:
        raise RuntimeError(f"No rain textures in {RAIN_TEXTURE_DIR}")

    _ensure_dir(OUT_DIR)
    _ensure_dir(DEPTH_DIR)

    if not SKIP_DEPTH_ESTIMATION:
        run_depth_estimation()

    with tqdm(total=len(haze_images), desc="(2/2) Compositing rain on haze") as pbar:
        for haze_path in haze_images:
            try:
                haze_rgb = _read_rgb(haze_path)

                clean_name = haze_path.stem + haze_path.suffix
                depth_path = (DEPTH_DIR / clean_name).with_suffix(".png")
                if not depth_path.exists():
                    depth_path = (DEPTH_DIR / haze_path.name).with_suffix(".png")

                depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
                if depth is None:
                    print(f"⚠️  Skip {haze_path.name}: missing depth map")
                    pbar.update(1)
                    continue

                depth01 = (depth.astype(np.float32) / 255.0)

                rain_tex_path = random.choice(rain_files)
                rain_tex = load_rain_texture_any(rain_tex_path)
                rain_patch = crop_or_tile_texture(rain_tex, haze_rgb.shape[0], haze_rgb.shape[1])

                comp = add_depth_aware_rain(haze_rgb, depth01, rain_patch)

                _write_bgr(OUT_DIR / haze_path.name, comp)
            except Exception as e:
                print(f"⚠️  Skip {haze_path.name}: {e}")
            pbar.update(1)


if __name__ == "__main__":
    main()