# Rendering split: host pipeline vs S600 BPU subgraphs

FoundationPose cannot be compiled end-to-end to HBM. The BPU adaptation is a
subgraph replacement: keep the geometry/rendering/control-flow pipeline on the
host and run the two static neural nets on S600.

## Why the whole pipeline is not a BPU graph

FoundationPose includes:

- dynamic Python registration/tracking loops (`estimater.py`);
- mesh loading, centering, diameter/symmetry handling, rotation grids;
- depth preprocessing and depth→XYZ map generation;
- **nvdiffrast CUDA/OpenGL rasterization** to render pose hypotheses;
- kornia crop/warp and bounding-window geometry;
- `mycpp.cluster_poses` and other custom/native utilities;
- pose-delta decoding, SE(3) composition, candidate sort/top-k/tournament.

These are not static neural-network graphs and contain CUDA/custom ops and Python
control flow. Trying to export them to ONNX/HBM would block the project without
solving the most measurable first milestone.

## What runs on BPU

### RefineNet

Input `A,B`: rendered and observed crop tensors, NCHW float32,
`(N, 6, 160, 160)`.

- channels 0..2: RGB
- channels 3..5: XYZ map

Output raw deltas:

- `trans (N,3)` — decode with upstream `trans_rep` (default `tracknet`: tanh ×
  `trans_normalizer`; `normalize_xyz` may scale by mesh diameter).
- `rot (N,3|6)` — axis-angle (3, default) or 6D; then convert to rotation matrix.

Host still performs the decode and SE(3) composition.

### ScoreNetMultiPair

Input `A,B`: candidate crop pairs, fixed `L` baked into the graph. The default
contract/export defaults include the historical `score_net_L20`, with
`A,B=(20,6,160,160)` and output `score_logit=(1,20)`, plus the current strict
L32 board target. Deployability must be validated per board/runtime. On the
current local S600/HBRT state, L20 needs the one-core preload shim to bypass
HBRT 4.7.5's pre-scheduling cross-core IOVA check, while `score_net_L32` loads
both normally and with the same preload policy.

Host still performs the iterative best-pair tournament / argmax and all candidate
bookkeeping. Because ScoreNet attends across the fixed `L` group, semantic BPU
validation requires exactly that many candidates; padding a shorter set is
load/control-flow smoke only.

## Hybrid dataflow

```
RGB/depth/K + mesh + previous pose
       │
       ├─ host: depth filter + depth→XYZ
       ├─ host: render pose hypothesis via nvdiffrast
       ├─ host: crop/warp rendered + observed tensors to 160x160
       ├─ host: concat rgb + xyz -> A/B
       │
       ├─ BPU: RefineNet(A,B) -> trans,rot
       │
       ├─ host: decode deltas + compose pose
       │        (repeat refine iterations)
       │
       ├─ host: render candidate set + crop/concat
       ├─ BPU: ScoreNetMultiPair(A,B,L) -> score_logit
       └─ host: select best candidate / update tracker
```

## Future embedded-rendering options (after subgraphs are stable)

Only after RefineNet/ScoreNet HBMs are verified and benchmarked should we assess
fully embedded alternatives, e.g.:

- pre-rendered template banks for objects with limited viewpoint coverage;
- lightweight CPU rasterizer for small meshes;
- replacing nvdiffrast with a platform-specific GPU/Vulkan/OpenGL path;
- approximating ScoreNet candidate generation with a learned template matcher.

Each is a separate research/engineering task and should be evaluated by end-to-
end ADD/ADD-S accuracy and frame-time, not just raw BPU latency.
