from .base_trainer import BaseTrainer
from .da_trainers import BaselineTrainer, DANNTrainer, CDANTrainer

_TRAINERS = {
    "baseline": BaselineTrainer,
    "dann": DANNTrainer,
    "cdan": CDANTrainer,
}


def build_trainer(cfg, meta, device):
    name = cfg["MODEL"]["METHOD"].lower()
    if name not in _TRAINERS:
        raise ValueError(f"Unknown method '{name}'")
    return _TRAINERS[name](cfg, meta, device)


__all__ = ["BaseTrainer", "BaselineTrainer", "DANNTrainer",
           "CDANTrainer", "build_trainer"]
