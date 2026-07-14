from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class BaseLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_cfg = cfg["DATA"]
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]

    def _read_raw(self):
        raise NotImplementedError

    def _normalize(self, X_src, X_tgt):
        raise NotImplementedError

    def _split_subjects(self, s):
        subs = np.unique(s)
        subs.sort()
        k = self.data_cfg["NUM_SOURCE_SUBJECTS"]
        src_subjects = subs[:k]
        tgt_subjects = subs[k:]
        src_mask = np.isin(s, src_subjects)
        tgt_mask = np.isin(s, tgt_subjects)
        return src_mask, tgt_mask, src_subjects, tgt_subjects

    def _class_weights(self, y_src, device):
        cnt = Counter(y_src.tolist())
        w = [len(y_src) / (self.num_classes * cnt[i]) for i in range(self.num_classes)]
        return torch.tensor(w, dtype=torch.float32, device=device)

    def build(self, device):
        X, y, s = self._read_raw()
        src_mask, tgt_mask, src_subs, tgt_subs = self._split_subjects(s)

        X_src, y_src = X[src_mask], y[src_mask]
        X_tgt, y_tgt = X[tgt_mask], y[tgt_mask]
        X_src, X_tgt = self._normalize(X_src, X_tgt)

        batch = self.data_cfg["BATCH_SIZE"]

        src_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_src).float(),
                          torch.from_numpy(y_src).long()),
            batch_size=batch, shuffle=True, drop_last=True,
        )
        tgt_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tgt).float()),
            batch_size=batch, shuffle=True, drop_last=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tgt).float(),
                          torch.from_numpy(y_tgt).long()),
            batch_size=batch, shuffle=False,
        )

        meta = {
            "in_dim": X_src.shape[1],
            "class_weights": self._class_weights(y_src, device),
            "src_subjects": src_subs.tolist(),
            "tgt_subjects": tgt_subs.tolist(),
            "n_src": len(y_src),
            "n_tgt": len(y_tgt),
        }
        return src_loader, tgt_loader, test_loader, meta
