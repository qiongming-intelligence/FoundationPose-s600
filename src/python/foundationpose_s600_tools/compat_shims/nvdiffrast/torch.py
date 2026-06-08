class RasterizeCudaContext:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('nvdiffrast CUDA renderer is unavailable on this S600 board')
class RasterizeGLContext:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('nvdiffrast GL renderer is unavailable on this S600 board')

def rasterize(*args, **kwargs):
    raise RuntimeError('nvdiffrast rasterize is unavailable on this S600 board')

def interpolate(*args, **kwargs):
    raise RuntimeError('nvdiffrast interpolate is unavailable on this S600 board')

def texture(*args, **kwargs):
    raise RuntimeError('nvdiffrast texture is unavailable on this S600 board')

def antialias(*args, **kwargs):
    raise RuntimeError('nvdiffrast antialias is unavailable on this S600 board')
