"""Debug / intermediate-tensor capture for FoundationPose S600 verification.

Use :mod:`foundationpose_s600_tools.debug.dump_intermediates` to wrap upstream
RefineNet / ScoreNet predictor modules and dump real A/B tensors for HBM parity
checks and calibration. Random inputs are invalid for these nets because of the
XYZ-map distribution.
"""
