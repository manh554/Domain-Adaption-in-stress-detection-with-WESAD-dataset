import os
import warnings
import glob
import pickle
from math import gcd

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "ACC_x_Mean", "ACC_x_Std", "ACC_x_Max", "ACC_x_PeakFreq", "ACC_x_AbsInt",
    "ACC_y_Mean", "ACC_y_Std", "ACC_y_Max", "ACC_y_PeakFreq", "ACC_y_AbsInt",
    "ACC_z_Mean", "ACC_z_Std", "ACC_z_Max", "ACC_z_PeakFreq", "ACC_z_AbsInt",
    "ACC_Mag_Mean", "ACC_Mag_Std",
    "EMG_Mean", "EMG_Std", "EMG_Max", "EMG_PeakFreq", "EMG_MeanFreq", "EMG_MedianFreq",
    "EDA_Mean", "EDA_Std", "EDA_Min", "EDA_Max", "EDA_Slope", "EDA_DynRange",
    "EDA_SCL_Mean", "EDA_SCL_Std", "EDA_SCR_Peaks", "EDA_SCR_Amp_Mean", "EDA_SCR_Area",
    "TEMP_Mean", "TEMP_Std", "TEMP_Min", "TEMP_Max", "TEMP_Slope", "TEMP_DynRange",
    "RSP_Rate_Mean", "RSP_Rate_Std", "RSP_Inhale_Mean", "RSP_Exhale_Mean",
    "RSP_IE_Ratio", "RSP_Stretch", "RSP_Power", "RSP_Breathing_Vol",
    "ECG_HR_Mean", "ECG_HR_Std",
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_TINN", "HRV_HTI",
    "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_VLF", "HRV_TP", "HRV_SampEn",
]

HRV_KEYS = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_TINN",
            "HRV_HTI", "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_VLF", "HRV_TP", "HRV_SampEn"]


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _subject_id(d, path):
    raw = d.get("subject", os.path.splitext(os.path.basename(path))[0])
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return int(digits) if digits else -1


def _fft_feats(sig, fs):
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(np.fft.rfft(x))
    total = mag.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    peak = float(freqs[np.argmax(mag)])
    meanf = float((freqs * mag).sum() / total)
    cum = np.cumsum(mag)
    medf = float(freqs[np.searchsorted(cum, cum[-1] / 2.0)])
    return peak, meanf, medf


def _stats(sig):
    s = np.asarray(sig, dtype=np.float64)
    return float(s.mean()), float(s.std()), float(s.min()), float(s.max())


def _slope(sig, fs):
    s = np.asarray(sig, dtype=np.float64)
    t = np.arange(len(s)) / fs
    if len(s) < 2:
        return 0.0
    return float(np.polyfit(t, s, 1)[0])


def _acc_feats(axis, fs, prefix, out):
    mean, std, _, mx = _stats(axis)
    peak, _, _ = _fft_feats(axis, fs)
    out[f"{prefix}_Mean"] = mean
    out[f"{prefix}_Std"] = std
    out[f"{prefix}_Max"] = mx
    out[f"{prefix}_PeakFreq"] = peak
    out[f"{prefix}_AbsInt"] = float(np.abs(axis).sum() / fs)


def _emg_feats(emg, fs, out):
    mean, std, _, mx = _stats(emg)
    peak, meanf, medf = _fft_feats(emg, fs)
    out.update({"EMG_Mean": mean, "EMG_Std": std, "EMG_Max": mx,
                "EMG_PeakFreq": peak, "EMG_MeanFreq": meanf, "EMG_MedianFreq": medf})


def _eda_feats(eda, fs, out, nk):
    mean, std, mn, mx = _stats(eda)
    out.update({"EDA_Mean": mean, "EDA_Std": std, "EDA_Min": mn, "EDA_Max": mx,
                "EDA_Slope": _slope(eda, fs), "EDA_DynRange": mx - mn})
    scl_mean = scl_std = scr_peaks = scr_amp = scr_area = 0.0
    try:
        sig, info = nk.eda_process(np.asarray(eda, dtype=np.float64), sampling_rate=fs)
        tonic = sig["EDA_Tonic"].values
        scl_mean, scl_std = float(np.mean(tonic)), float(np.std(tonic))
        amps = np.asarray(info.get("SCR_Amplitude", []), dtype=np.float64)
        amps = amps[~np.isnan(amps)] if amps.size else amps
        scr_peaks = float(amps.size)
        scr_amp = float(amps.mean()) if amps.size else 0.0
        scr_area = float(amps.sum()) if amps.size else 0.0
    except Exception:
        pass
    out.update({"EDA_SCL_Mean": scl_mean, "EDA_SCL_Std": scl_std,
                "EDA_SCR_Peaks": scr_peaks, "EDA_SCR_Amp_Mean": scr_amp,
                "EDA_SCR_Area": scr_area})


def _temp_feats(temp, fs, out):
    mean, std, mn, mx = _stats(temp)
    out.update({"TEMP_Mean": mean, "TEMP_Std": std, "TEMP_Min": mn, "TEMP_Max": mx,
                "TEMP_Slope": _slope(temp, fs), "TEMP_DynRange": mx - mn})


def _rsp_feats(rsp, fs, out, nk):
    vals = {k: 0.0 for k in ["RSP_Rate_Mean", "RSP_Rate_Std", "RSP_Inhale_Mean",
                             "RSP_Exhale_Mean", "RSP_IE_Ratio", "RSP_Stretch",
                             "RSP_Power", "RSP_Breathing_Vol"]}
    try:
        sig, info = nk.rsp_process(np.asarray(rsp, dtype=np.float64), sampling_rate=fs)
        rate = sig["RSP_Rate"].values
        amp = sig["RSP_Amplitude"].values
        vals["RSP_Rate_Mean"] = float(np.nanmean(rate))
        vals["RSP_Rate_Std"] = float(np.nanstd(rate))
        vals["RSP_Stretch"] = float(np.nanmean(amp))
        vals["RSP_Power"] = float(np.nanvar(np.asarray(rsp, dtype=np.float64)))
        vals["RSP_Breathing_Vol"] = float(np.nansum(np.abs(amp)) / fs)
        peaks = np.asarray(info.get("RSP_Peaks", []), dtype=float)
        troughs = np.asarray(info.get("RSP_Troughs", []), dtype=float)
        peaks = peaks[~np.isnan(peaks)]
        troughs = troughs[~np.isnan(troughs)]
        if peaks.size and troughs.size:
            inh, exh = [], []
            for p in peaks:
                prev = troughs[troughs < p]
                nxt = troughs[troughs > p]
                if prev.size:
                    inh.append((p - prev[-1]) / fs)
                if nxt.size:
                    exh.append((nxt[0] - p) / fs)
            im = float(np.mean(inh)) if inh else 0.0
            em = float(np.mean(exh)) if exh else 0.0
            vals["RSP_Inhale_Mean"] = im
            vals["RSP_Exhale_Mean"] = em
            vals["RSP_IE_Ratio"] = float(im / em) if em > 0 else 0.0
    except Exception:
        pass
    out.update(vals)


def _ecg_hrv_feats(ecg, fs, out, nk):
    out["ECG_HR_Mean"] = 0.0
    out["ECG_HR_Std"] = 0.0
    for k in HRV_KEYS:
        out[k] = 0.0
    try:
        sig, info = nk.ecg_process(np.asarray(ecg, dtype=np.float64), sampling_rate=fs)
        hr = sig["ECG_Rate"].values
        out["ECG_HR_Mean"] = float(np.nanmean(hr))
        out["ECG_HR_Std"] = float(np.nanstd(hr))
        hrv = nk.hrv(info, sampling_rate=fs, show=False)
        for k in HRV_KEYS:
            if k in hrv.columns:
                v = hrv[k].values[0]
                out[k] = float(v) if v == v else 0.0
    except Exception:
        pass


def _window_features(win, fs, nk):
    out = {}
    ax, ay, az = win["ACC"][:, 0], win["ACC"][:, 1], win["ACC"][:, 2]
    _acc_feats(ax, fs, "ACC_x", out)
    _acc_feats(ay, fs, "ACC_y", out)
    _acc_feats(az, fs, "ACC_z", out)
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    out["ACC_Mag_Mean"], out["ACC_Mag_Std"] = float(mag.mean()), float(mag.std())
    _emg_feats(win["EMG"], fs, out)
    _eda_feats(win["EDA"], fs, out, nk)
    _temp_feats(win["Temp"], fs, out)
    _rsp_feats(win["Resp"], fs, out, nk)
    _ecg_hrv_feats(win["ECG"], fs, out, nk)
    return out


class WESADFeatureExtractor:
    def __init__(self, cfg):
        p = cfg["PREPROCESS"]
        self.raw_dir = p["RAW_DIR"]
        self.fs = p["ORIG_FS"]
        self.window_sec = p.get("FEAT_WINDOW_SEC", 60.0)
        self.step_sec = p.get("FEAT_STEP_SEC", 1.0)
        self.purity = p["LABEL_PURITY"]
        self.keep = list(p["KEEP_LABELS"])
        self.out_csv = cfg["DATA"]["CSV_PATH"]
        self.win = int(round(self.window_sec * self.fs))
        self.step = int(round(self.step_sec * self.fs))

    def _chest_arrays(self, chest):
        return {
            "ACC": np.asarray(chest["ACC"], dtype=np.float64),
            "ECG": np.asarray(chest["ECG"], dtype=np.float64).ravel(),
            "EMG": np.asarray(chest["EMG"], dtype=np.float64).ravel(),
            "EDA": np.asarray(chest["EDA"], dtype=np.float64).ravel(),
            "Temp": np.asarray(chest["Temp"], dtype=np.float64).ravel(),
            "Resp": np.asarray(chest["Resp"], dtype=np.float64).ravel(),
        }

    def run(self):
        import neurokit2 as nk

        paths = sorted(glob.glob(os.path.join(self.raw_dir, "S*.pkl")))
        if not paths:
            raise FileNotFoundError(f"No S*.pkl found in {self.raw_dir}")

        rows, labels, sids = [], [], []
        for path in paths:
            d = _load_pkl(path)
            ch = self._chest_arrays(d["signal"]["chest"])
            y = np.asarray(d["label"]).astype(int).ravel()
            sid = _subject_id(d, path)
            n = min(len(ch["ECG"]), len(y))

            cnt = 0
            for start in range(0, n - self.win + 1, self.step):
                lab = y[start:start + self.win]
                vals, counts = np.unique(lab, return_counts=True)
                maj = int(vals[counts.argmax()])
                if maj not in self.keep or counts.max() / self.win < self.purity:
                    continue
                win = {
                    "ACC": ch["ACC"][start:start + self.win],
                    "ECG": ch["ECG"][start:start + self.win],
                    "EMG": ch["EMG"][start:start + self.win],
                    "EDA": ch["EDA"][start:start + self.win],
                    "Temp": ch["Temp"][start:start + self.win],
                    "Resp": ch["Resp"][start:start + self.win],
                }
                rows.append(_window_features(win, self.fs, nk))
                labels.append(maj)
                sids.append(sid)
                cnt += 1
            print(f"  {os.path.basename(path)} (sid={sid}): {cnt} windows")

        df = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS).fillna(0.0)
        df["label"] = labels
        df["sid"] = sids

        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        df.to_csv(self.out_csv, index=False)
        print(f">>> saved features={df.shape} -> {self.out_csv}")
        return df


def extract_features(cfg):
    return WESADFeatureExtractor(cfg).run()
