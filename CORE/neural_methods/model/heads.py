import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int,
                 hidden: int = 0, dropout: float = 0.3):
        super().__init__()
        if hidden and hidden > 0:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
