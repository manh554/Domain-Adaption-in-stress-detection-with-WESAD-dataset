import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

from ..model import build_backbone, Classifier


class BaseTrainer:
    METHOD = "base"

    def __init__(self, cfg, meta, device):
        self.cfg = cfg
        self.meta = meta
        self.device = device
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]
        self.epochs = cfg["TRAIN"]["EPOCHS"]

        mcfg = cfg["MODEL"]
        self.backbone = build_backbone(mcfg["BACKBONE"], meta["in_dim"]).to(device)
        self.classifier = Classifier(
            self.backbone.out_dim, self.num_classes,
            hidden=mcfg.get("HEAD_HIDDEN", 0),
            dropout=mcfg.get("HEAD_DROPOUT", 0.3),
        ).to(device)

        self.ce = nn.CrossEntropyLoss(weight=meta["class_weights"])

        self.extra_params = []
        self._setup_da()

        params = (list(self.backbone.parameters())
                  + list(self.classifier.parameters())
                  + self.extra_params)
        opt_name = cfg["TRAIN"].get("OPTIMIZER", "adamw").lower()
        wd = cfg["TRAIN"].get("WEIGHT_DECAY", 0.0)
        OptCls = {"adam": optim.Adam, "adamw": optim.AdamW}[opt_name]
        self.opt = OptCls(params, lr=cfg["TRAIN"]["LR"], weight_decay=wd)

        os.makedirs(cfg["LOG"]["CKPT_DIR"], exist_ok=True)
        self.ckpt = os.path.join(cfg["LOG"]["CKPT_DIR"], f"{cfg['MODEL']['NAME']}.pth")
        self.history = {"cls": [], "da": [], "lambda": [], "tgt_acc": []}

    def _setup_da(self):
        pass

    def _train_step(self, batch_src, batch_tgt, lambd):
        raise NotImplementedError

    def _lambda(self, epoch):
        tcfg = self.cfg["TRAIN"]
        warmup = tcfg.get("LAMBDA_WARMUP_EPOCHS", 0)
        cap = tcfg.get("LAMBDA_MAX", 1.0)
        if epoch < warmup:
            return 0.0
        denom = max(1, self.epochs - warmup)
        p = (epoch - warmup) / denom
        return (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * cap

    def _train_mode(self, flag=True):
        self.backbone.train(flag)
        self.classifier.train(flag)

    def train(self, src_loader, tgt_loader, test_loader):
        best_acc, sel = 0.0, self.cfg["TRAIN"].get("MODEL_SELECTION", "last")

        for epoch in range(self.epochs):
            self._train_mode(True)
            lambd = self._lambda(epoch)
            tgt_iter = iter(tgt_loader)
            cls_sum, da_sum, n = 0.0, 0.0, 0

            for batch_src in src_loader:
                try:
                    batch_tgt = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    batch_tgt = next(tgt_iter)

                out = self._train_step(batch_src, batch_tgt, lambd)
                self.opt.zero_grad()
                out["loss"].backward()
                self.opt.step()

                cls_sum += out["cls"]
                da_sum += out["da"]
                n += 1

            acc = self.evaluate(test_loader)
            self.history["cls"].append(cls_sum / n)
            self.history["da"].append(da_sum / n)
            self.history["lambda"].append(lambd)
            self.history["tgt_acc"].append(acc)

            print(f"Epoch {epoch + 1:02d} | cls {cls_sum / n:.3f} | "
                  f"da {da_sum / n:.3f} | lambda {lambd:.3f} | tgt_acc {acc:.2f}%")

            if sel == "best_target" and acc > best_acc:
                best_acc = acc
                self.save()

        if sel != "best_target":
            self.save()
        else:
            print(f">>> BEST TARGET ACC: {best_acc:.2f}%")
        return self.history

    @torch.no_grad()
    def evaluate(self, loader):
        self._train_mode(False)
        preds, gts = [], []
        for x, y in loader:
            out = self.classifier(self.backbone(x.to(self.device))).argmax(1)
            preds.extend(out.cpu().numpy())
            gts.extend(y.numpy())
        return accuracy_score(gts, preds) * 100

    @torch.no_grad()
    def test(self, loader, target_names=None):
        self.load()
        self._train_mode(False)
        preds, gts = [], []
        for x, y in loader:
            out = self.classifier(self.backbone(x.to(self.device))).argmax(1)
            preds.extend(out.cpu().numpy())
            gts.extend(y.numpy())
        print("\nClassification Report:")
        print(classification_report(gts, preds, target_names=target_names, digits=4))
        return np.array(gts), np.array(preds)

    def save(self):
        torch.save({
            "backbone": self.backbone.state_dict(),
            "classifier": self.classifier.state_dict(),
        }, self.ckpt)

    def load(self):
        state = torch.load(self.ckpt, map_location=self.device)
        self.backbone.load_state_dict(state["backbone"])
        self.classifier.load_state_dict(state["classifier"])

    @torch.no_grad()
    def extract_features(self, loader, max_samples=2000):
        self._train_mode(False)
        feats, labels = [], []
        for x, y in loader:
            feats.append(self.backbone(x.to(self.device)).cpu().numpy())
            labels.append(y.numpy())
            if sum(len(f) for f in feats) >= max_samples:
                break
        X = np.concatenate(feats)[:max_samples]
        y = np.concatenate(labels)[:max_samples]
        return X, y
