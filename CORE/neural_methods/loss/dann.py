import torch
import torch.nn as nn

from ..model.grl import GRL


class DANNLoss(nn.Module):
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.D = discriminator
        self.grl = GRL()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, f_s, f_t, lambd: float = 1.0):
        f = torch.cat([f_s, f_t], dim=0)
        f = self.grl(f, lambd)
        logits = self.D(f)

        d_labels = torch.cat([
            torch.ones(f_s.size(0), 1, device=f.device),
            torch.zeros(f_t.size(0), 1, device=f.device),
        ], dim=0)
        return self.bce(logits, d_labels)
