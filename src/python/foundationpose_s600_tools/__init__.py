"""FoundationPose → S600 BPU adaptation tooling.

This package provides the static-subgraph export, conversion, and hybrid-runtime
glue for running FoundationPose's two learning networks (RefineNet and
ScoreNetMultiPair) on the Horizon / D-Robotics S600 BPU. The surrounding pipeline
(rendering, depth/XYZ, pose composition) stays on CPU/GPU; see docs/rendering_split.md.
"""

__all__ = ["export", "convert"]
