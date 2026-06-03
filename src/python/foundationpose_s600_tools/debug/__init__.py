"""Debug / intermediate-tensor capture for FoundationPose S600 verification.

Placeholder for plan §6/§8: capture the exact A/B tensors fed to RefineNet /
ScoreNetMultiPair from a running FoundationPose pipeline, for (a) PyTorch-vs-HBM
parity checks and (b) real calibration data (see configs/calibration/README.md).
Random inputs are invalid for these nets because of the XYZ-map distribution.
"""
