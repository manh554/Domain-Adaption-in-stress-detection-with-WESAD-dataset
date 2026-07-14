import torch
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from ..model import DomainDiscriminator
from ..loss import DANNLoss, CDANLoss


class BaselineTrainer(BaseTrainer):
    METHOD = "baseline"

    def _train_step(self, batch_src, batch_tgt, lambd):
        xs, ys = batch_src[0].to(self.device), batch_src[1].to(self.device)
        loss = self.ce(self.classifier(self.backbone(xs)), ys)
        return {"loss": loss, "cls": loss.item(), "da": 0.0}


class DANNTrainer(BaseTrainer):
    METHOD = "dann"

    def _setup_da(self):
        dcfg = self.cfg["MODEL"]
        self.D = DomainDiscriminator(
            self.backbone.out_dim, hidden=dcfg.get("DISC_HIDDEN", 1024),
        ).to(self.device)
        self.dann = DANNLoss(self.D).to(self.device)
        self.extra_params = list(self.D.parameters())

    def _train_step(self, batch_src, batch_tgt, lambd):
        xs, ys = batch_src[0].to(self.device), batch_src[1].to(self.device)
        xt = batch_tgt[0].to(self.device)

        fs, ft = self.backbone(xs), self.backbone(xt)
        loss_cls = self.ce(self.classifier(fs), ys)
        loss_da = self.dann(fs, ft, lambd)
        loss = loss_cls + loss_da
        return {"loss": loss, "cls": loss_cls.item(), "da": loss_da.item()}


class CDANTrainer(BaseTrainer):
    METHOD = "cdan"

    def _setup_da(self):
        mcfg = self.cfg["MODEL"]
        entropy = mcfg.get("ENTROPY_CONDITIONING", True)
        randomized = mcfg.get("RANDOMIZED", False)
        rdim = mcfg.get("RANDOMIZED_DIM", 1024)

        disc_in = rdim if randomized else self.backbone.out_dim * self.num_classes
        self.D = DomainDiscriminator(
            disc_in, hidden=mcfg.get("DISC_HIDDEN", 1024),
        ).to(self.device)
        self.cdan = CDANLoss(
            self.D, num_classes=self.num_classes,
            features_dim=self.backbone.out_dim,
            entropy_conditioning=entropy,
            randomized=randomized, randomized_dim=rdim,
        ).to(self.device)
        self.extra_params = list(self.D.parameters())

    def _train_step(self, batch_src, batch_tgt, lambd):
        xs, ys = batch_src[0].to(self.device), batch_src[1].to(self.device)
        xt = batch_tgt[0].to(self.device)

        fs, ft = self.backbone(xs), self.backbone(xt)
        ps, pt = self.classifier(fs), self.classifier(ft)
        loss_cls = self.ce(ps, ys)
        loss_da = self.cdan(
            F.softmax(ps, dim=1), fs, F.softmax(pt, dim=1), ft, lambd,
        )
        loss = loss_cls + loss_da
        return {"loss": loss, "cls": loss_cls.item(), "da": loss_da.item()}
