import os
import glob
import pickle
from math import gcd

import numpy as np
from scipy.signal import resample_poly


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _subject_id(d, path):
    raw = d.get("subject", os.path.splitext(os.path.basename(path))[0])
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return int(digits) if digits else -1


class WESADPreprocessor:
    def __init__(self, cfg):
        p = cfg["PREPROCESS"]
        self.raw_dir = p["RAW_DIR"]
        self.orig_fs = p["ORIG_FS"]
        self.target_fs = p["TARGET_FS"]
        self.window_sec = p["WINDOW_SEC"]
        self.step_sec = p["STEP_SEC"]
        self.purity = p["LABEL_PURITY"]
        self.keep = list(p["KEEP_LABELS"])
        self.channels = list(p["CHEST_CHANNELS"])
        self.out_x = cfg["DATA"]["X_PATH"]
        self.out_y = cfg["DATA"]["Y_PATH"]
        self.out_s = cfg["DATA"]["S_PATH"]

        g = gcd(self.target_fs, self.orig_fs)
        self.up = self.target_fs // g
        self.down = self.orig_fs // g
        self.win = int(round(self.window_sec * self.orig_fs))
        self.step = int(round(self.step_sec * self.orig_fs))
        self.out_len = int(round(self.window_sec * self.target_fs))

    def _stack_chest(self, chest):
        cols = []
        for ch in self.channels:
            sig = np.asarray(chest[ch], dtype=np.float32)
            if sig.ndim == 1:
                sig = sig[:, None]
            cols.append(sig)
        return np.concatenate(cols, axis=1)

    def _fix_len(self, seg):
        n = seg.shape[0]
        if n == self.out_len:
            return seg
        if n > self.out_len:
            return seg[:self.out_len]
        pad = np.repeat(seg[-1:], self.out_len - n, axis=0)
        return np.concatenate([seg, pad], axis=0)

    def _window_subject(self, X, y):
        Xs, ys = [], []
        for start in range(0, len(X) - self.win + 1, self.step):
            lab = y[start:start + self.win]
            vals, counts = np.unique(lab, return_counts=True)
            maj = int(vals[counts.argmax()])
            if maj not in self.keep:
                continue
            if counts.max() / self.win < self.purity:
                continue
            seg = X[start:start + self.win]
            seg_rs = resample_poly(seg, self.up, self.down, axis=0)
            seg_rs = self._fix_len(seg_rs)
            Xs.append(seg_rs.T.astype(np.float32))
            ys.append(maj)
        return Xs, ys

    def run(self):
        paths = sorted(glob.glob(os.path.join(self.raw_dir, "S*.pkl")))
        if not paths:
            raise FileNotFoundError(f"No S*.pkl found in {self.raw_dir}")

        X_all, y_all, s_all = [], [], []
        for path in paths:
            d = _load_pkl(path)
            chest = d["signal"]["chest"]
            X = self._stack_chest(chest)
            y = np.asarray(d["label"]).astype(int).ravel()
            n = min(len(X), len(y))
            X, y = X[:n], y[:n]
            sid = _subject_id(d, path)

            Xs, ys = self._window_subject(X, y)
            X_all.extend(Xs)
            y_all.extend(ys)
            s_all.extend([sid] * len(ys))
            print(f"  {os.path.basename(path)} (sid={sid}): {len(ys)} windows")

        X_all = np.stack(X_all)
        y_all = np.asarray(y_all, dtype=np.int64)
        s_all = np.asarray(s_all, dtype=np.int64)

        for pth in (self.out_x, self.out_y, self.out_s):
            os.makedirs(os.path.dirname(pth) or ".", exist_ok=True)
        np.save(self.out_x, X_all)
        np.save(self.out_y, y_all)
        np.save(self.out_s, s_all)

        print(f">>> saved X={X_all.shape} -> {self.out_x}")
        print(f">>> saved y={y_all.shape} -> {self.out_y}")
        print(f">>> saved s={s_all.shape} -> {self.out_s}")
        return X_all, y_all, s_all
