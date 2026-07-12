# Code audit and resolution record

## Decision

The manuscript is now treated as the authoritative source for the released
synthesis pipeline. Historical constants and formulas remain available only in
`archive/historical_pipeline/`.

## Historical mismatch and current resolution

| Item | Historical implementation | Canonical implementation |
|---|---|---|
| Entry point | Several overlapping scripts | `generate_marinerain.py` only |
| Paths | Hard-coded Windows drive paths | Required command-line directory arguments |
| Gamma | `2.5` plus hotspot suppression | Paper equation with default `gamma=3.5` |
| Relative depth | Normalized disparity, inverted in later stages | Explicitly computed as `1 - per-image normalized disparity` |
| Haze/background attenuation | Linear clipped depth blend | `T*B + L*(1-T)`, with `T=exp(-beta*d)` |
| Rain attenuation | Separate `alpha*R*exp(-beta*d)` stage | `T*R_gc`, using the same `T` as the background |
| Beta | Usually sampled from `[0.9,1.1]` | Sampled from `[1.0,1.2]` by default |
| Atmospheric light | Fixed `200/255` | Sampled from `[0.8,1.0]` by default |
| Reproducibility record | Seed in some scripts | Seed plus per-sample CSV parameters and source mapping |
| Environment | No environment file | Pinned reference `environments.txt` |

## Canonical equation

The implementation in `marinerain_pipeline.py` uses normalized floating-point
images and computes:

```text
R_gc = R_orig ^ gamma
T    = exp(-beta * d)
O    = T * B + L * (1 - T) + T * R_gc
```

No independent historical rain multiplier is applied in the canonical path.

## Completed scope

1. Chose the manuscript as parameter/formula authority.
2. Set the canonical Gamma default to `3.5`.
3. Unified background attenuation, airlight, and rain attenuation in one
   compositing function.
4. Set the reported `beta` and atmospheric-light sampling ranges.
5. Replaced machine-specific paths with command-line options.
6. Added one documented end-to-end entry point.
7. Added the requested pinned `environments.txt` based on repository imports.

## Explicitly excluded scope

At the user's request, this pass did not:

- add or verify third-party licenses;
- define a model-weight distribution/download workflow;
- compare generated output against original samples or historical results.

The final verification is consequently static and structural rather than an
empirical reproduction claim.
