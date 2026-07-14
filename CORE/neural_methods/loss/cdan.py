import numpy as np
import torch
import torch.nn as nn

from ..model.grl import GRL


def _entropy(prob, eps: float = 1e-6):
    return -torch.sum(prob * torch.log(prob + eps), dim=1)


class CDANLoss(nn.Module):
    def __init__(self, discriminator: nn.Module,
                 num_classes: int, features_dim: int,
                 entropy_conditioning: bool = True,
                 randomized: bool = False, randomized_dim: int = 1024):
        super().__init__()
        self.D = discriminator
        self.grl = GRL()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.entropy_conditioning = entropy_conditioning
        self.randomized = randomized

        if randomized:
            self.register_buffer("Rf", torch.randn(features_dim, randomized_dim))
            self.register_buffer("Rp", torch.randn(num_classes, randomized_dim))
            self.out_dim = randomized_dim
        else:
            self.out_dim = features_dim * num_classes

    def _map(self, prob, feat):
        if self.randomized:
            f = feat @ self.Rf
            p = prob @ self.Rp
            return f * p / np.sqrt(self.out_dim)
        return torch.bmm(prob.unsqueeze(2), feat.unsqueeze(1)).view(feat.size(0), -1)

    def forward(self, prob_s, f_s, prob_t, f_t, lambd: float = 1.0):
        g_s = self._map(prob_s, f_s)
        g_t = self._map(prob_t, f_t)

        g = torch.cat([g_s, g_t], dim=0)
        g = self.grl(g, lambd)
        logits = self.D(g)

        d_labels = torch.cat([
            torch.ones(f_s.size(0), 1, device=g.device),
            torch.zeros(f_t.size(0), 1, device=g.device),
        ], dim=0)

        loss = self.bce(logits, d_labels).view(-1)

        if self.entropy_conditioning:
            prob = torch.cat([prob_s, prob_t], dim=0)
            w = torch.exp(-_entropy(prob))
            w = w / (w.mean() + 1e-6)
            loss = loss * w

        return loss.mean()
