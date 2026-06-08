"""Python hybrid runtime adapters (S600 BPU ⇆ FoundationPose loop).

These adapters feed host-generated A/B crop tensors to compiled RefineNet and
ScoreNet HBMs and return dict outputs matching the upstream PyTorch modules.
They support the legacy ``hrt_model_exec`` validation backend and the persistent
native C++/UCP backend that keeps HBMs loaded across inference calls.
"""

from foundationpose_s600_tools.runtime.hybrid import install_bpu_adapters
from foundationpose_s600_tools.runtime.persistent_bpu import PersistentBpuModelRunner, PersistentBpuSession, PersistentModelSpec
from foundationpose_s600_tools.runtime.refine_bpu import RefineNetBpu
from foundationpose_s600_tools.runtime.score_bpu import ScoreNetBpu, ScoreNetBpuL16, ScoreNetBpuL20, ScoreNetBpuL32

__all__ = [
    "PersistentBpuModelRunner",
    "PersistentBpuSession",
    "PersistentModelSpec",
    "RefineNetBpu",
    "ScoreNetBpu",
    "ScoreNetBpuL16",
    "ScoreNetBpuL20",
    "ScoreNetBpuL32",
    "install_bpu_adapters",
]
