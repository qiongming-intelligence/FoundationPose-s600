class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise ImportError('PyTorch3D renderer is unavailable in the S600 shim')
FoVPerspectiveCameras = PerspectiveCameras = RasterizationSettings = MeshRenderer = MeshRasterizer = BlendParams = SoftSilhouetteShader = HardPhongShader = PointLights = TexturesVertex = _Unavailable

def look_at_view_transform(*args, **kwargs):
    raise ImportError('PyTorch3D renderer is unavailable in the S600 shim')

def look_at_rotation(*args, **kwargs):
    raise ImportError('PyTorch3D renderer is unavailable in the S600 shim')
