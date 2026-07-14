import numpy as np
import pandas as pd

from .base_loader import BaseLoader


class WESADFeatureLoader(BaseLoader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._scaler_mean = None
        self._scaler_std = None

    def _read_raw(self):
        d = self.data_cfg
        df = pd.read_csv(d["CSV_PATH"]).fillna(0)

        s = df[d["SID_COL"]].values
        y = df[d["LABEL_COL"]].values
        X = df.drop(columns=[d["SID_COL"], d["LABEL_COL"]]).values.astype(np.float32)

        classes = np.sort(np.unique(y))
        remap = {c: i for i, c in enumerate(classes)}
        y = np.vectorize(remap.get)(y).astype(np.int64)
        return X, y, s

    def _normalize(self, X_src, X_tgt):
        self._scaler_mean = X_src.mean(axis=0, keepdims=True)
        self._scaler_std = X_src.std(axis=0, keepdims=True) + 1e-6
        Xs = (X_src - self._scaler_mean) / self._scaler_std
        Xt = (X_tgt - self._scaler_mean) / self._scaler_std
        return Xs.astype(np.float32), Xt.astype(np.float32)
