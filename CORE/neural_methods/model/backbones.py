import torch.nn as nn


class FeatureCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.out_dim = 256

    def forward(self, x):
        return self.net(x)


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x)


def build_backbone(name: str, in_dim: int, **kwargs):
    name = name.lower()
    if name == "cnn":
        return FeatureCNN(in_dim)
    if name == "mlp":
        return FeatureMLP(in_dim, **kwargs)
    raise ValueError(f"Unknown backbone '{name}'")
