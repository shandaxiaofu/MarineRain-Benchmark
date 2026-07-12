# Historical source organization map

The files below originated in `/Users/gaohan/Desktop/syn_monadepth`. Their
contents were copied byte-for-byte and are retained under
`archive/historical_pipeline/` for provenance.

| Original name | Archived name |
|---|---|
| `gamma.py` | `archive/historical_pipeline/01_gamma_correct_rain_layers.py` |
| `hazy_depthmap_Monodepth.py` | `archive/historical_pipeline/02_estimate_depth_and_generate_haze.py` |
| `merge_RF.py` | `archive/historical_pipeline/03_add_depth_aware_rain_to_haze.py` |
| `merge.py` | `archive/historical_pipeline/04_add_depth_aware_rain_only.py` |
| `hazy_depthmap_Monodepth_with_rain.py` | `archive/historical_pipeline/legacy_all_in_one_depth_haze_rain.py` |
| `api.py` | `evaluation/ship_detection_and_voc_ap.py` |
| `one.py` | `evaluation/debug_single_image_detection.py` |
| `bigsmallimg.py` | `tools/crop_marked_regions.py` |
| `utils/depth_decoder.py` | `utils/depth_decoder.py` |
| `utils/layers.py` | `utils/layers.py` |
| `utils/resnet_encoder.py` | `utils/resnet_encoder.py` |

The following files are newly authored canonical release files rather than
renamed historical sources:

- `generate_marinerain.py`
- `marinerain_pipeline.py`
- `README.md`
- `CODE_AUDIT.md`
- `environments.txt`
- `.gitignore`

The trivial historical scratch file `test.py` remains excluded.
