# Monodepth2 Model Weights

The MarineRain synthesis pipeline uses the Monodepth2
`mono+stereo_1024x320` model for single-image relative-depth estimation.

Place the pretrained model files in the following directory:

```text
models/
└── mono+stereo_1024x320/
    ├── encoder.pth
    └── depth.pth
```

Only the following two files are required:

- `encoder.pth`: pretrained ResNet encoder and input-resolution metadata;
- `depth.pth`: pretrained Monodepth2 depth decoder.

Pose-estimation weights such as `pose.pth` and `pose_encoder.pth` are not used
by the MarineRain synthesis pipeline.

To use a different local model directory, pass it explicitly:

```bash
python generate_marinerain.py \
  --clean-dir data/clean \
  --rain-dir data/rain_layers \
  --output-dir outputs/marinerain \
  --model-dir /path/to/monodepth2/model
```

Model weight files are excluded from Git through the repository `.gitignore`.
