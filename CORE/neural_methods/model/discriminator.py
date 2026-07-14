import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
