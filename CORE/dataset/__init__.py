from .base_loader import BaseLoader
from .wesad_raw_loader import WESADRawLoader
from .wesad_feature_loader import WESADFeatureLoader

_LOADERS = {
    "wesad_raw": WESADRawLoader,
    "wesad_feature": WESADFeatureLoader,
}


def build_loader(cfg):
    name = cfg["DATA"]["LOADER"].lower()
    if name not in _LOADERS:
        raise ValueError(f"Unknown loader '{name}'")
    return _LOADERS[name](cfg)


__all__ = ["BaseLoader", "WESADRawLoader", "WESADFeatureLoader", "build_loader"]
