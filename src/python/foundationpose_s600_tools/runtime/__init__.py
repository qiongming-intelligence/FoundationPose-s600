"""Python hybrid runtime adapters (S600 BPU ⇆ FoundationPose loop).

Placeholder for plan §6: thin adapters that feed host-generated A/B crop tensors
to the compiled RefineNet/ScoreNet HBM and return trans/rot/score to the upstream
estimater loop, so the BPU subgraph substitution can be validated against the
PyTorch baseline. Implemented after the C++ raw-tensor runner (plan §5) lands.
"""
