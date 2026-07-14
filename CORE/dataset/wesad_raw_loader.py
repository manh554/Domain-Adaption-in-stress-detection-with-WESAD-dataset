import numpy as np

from .base_loader import BaseLoader


class WESADRawLoader(BaseLoader):
    def _read_raw(self):
        d = self.data_cfg
        X = np.load(d["X_PATH"]).astype(np.float32)
        y = np.load(d["Y_PATH"])
        s = np.load(d["S_PATH"])

        mask = np.isin(y, [1, 2, 3])
        X, y, s = X[mask], y[mask], s[mask]
        y = np.vectorize({1: 0, 2: 1, 3: 2}.get)(y).astype(np.int64)

        if X.shape[2] < X.shape[1]:
            X = X.transpose(0, 2, 1)
        return X, y, s

    def _normalize(self, X_src, X_tgt):
        mean = X_src.mean(axis=(0, 2), keepdims=True)
        std = X_src.std(axis=(0, 2), keepdims=True) + 1e-6
        return (X_src - mean) / std, (X_tgt - mean) / std
