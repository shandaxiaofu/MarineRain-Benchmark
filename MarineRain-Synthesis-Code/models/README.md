# Model weights

Model weights are not included in this organized copy.

The synthesis scripts require the Monodepth2 `encoder.pth` and `depth.pth`
files in the model directory configured inside each historical script. The
original desktop archive contains six model variants, but only encoder/depth
weights are used by the synthesis path; pose and pose-encoder weights are not
needed for single-image inference.

Before publishing weights or download links, restore and review the official
Monodepth2 license and attribution. The copied network source explicitly states
that its license permits non-commercial use only.
