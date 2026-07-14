import copy

import yaml

DEFAULT = {
    "MODEL": {
        "NAME": "run",
        "METHOD": "baseline",
        "BACKBONE": "cnn",
        "NUM_CLASSES": 3,
        "HEAD_HIDDEN": 0,
        "HEAD_DROPOUT": 0.3,
        "DISC_HIDDEN": 1024,
        "ENTROPY_CONDITIONING": True,
        "RANDOMIZED": False,
        "RANDOMIZED_DIM": 1024,
    },
    "DATA": {
        "LOADER": "wesad_raw",
        "BATCH_SIZE": 64,
        "NUM_SOURCE_SUBJECTS": 10,

        # === NOTICE: đặt các file dữ liệu ở đâu ===
        # Mặc định tìm ngay trong thư mục chạy lệnh (thư mục gốc của toolbox).
        # - CNN (LOADER: wesad_raw) cần 3 file .npy dưới đây.
        # - MLP (LOADER: wesad_feature) cần file CSV_PATH.
        # Nếu để dữ liệu trong thư mục con, ví dụ ./dataset_files/, chỉ cần đổi
        # đường dẫn ở đây hoặc ghi đè trong file YAML, ví dụ:
        #     X_PATH: dataset_files/wesad_X_raw_70Hz.npy
        #     CSV_PATH: dataset_files/2.csv
        "X_PATH": "wesad_X_raw_70Hz.npy",
        "Y_PATH": "wesad_y_raw_labels.npy",
        "S_PATH": "wesad_s_raw_subjects.npy",
        "CSV_PATH": "2.csv",
        "SID_COL": "sid",
        "LABEL_COL": "label",
    },
    "TRAIN": {
        "EPOCHS": 60,
        "LR": 1e-3,
        "OPTIMIZER": "adamw",
        "WEIGHT_DECAY": 1e-3,
        "LAMBDA_MAX": 0.3,
        "LAMBDA_WARMUP_EPOCHS": 5,
        "MODEL_SELECTION": "last",
        "SEED": 42,
    },
    "PREPROCESS": {
        # === NOTICE: đặt dữ liệu WESAD gốc ở đâu ===
        # RAW_DIR trỏ tới thư mục chứa các file S2.pkl, S3.pkl, ... (định dạng
        # WESAD chính thức). Preprocessing đọc CHEST 8 kênh theo CHEST_CHANNELS,
        # resample ORIG_FS -> TARGET_FS, cắt window rồi ghi ra 3 file .npy
        # (đường dẫn lấy từ DATA.X_PATH / Y_PATH / S_PATH ở trên).
        "RAW_DIR": "WESAD",
        "ORIG_FS": 700,
        "TARGET_FS": 70,
        "WINDOW_SEC": 2.0,
        "STEP_SEC": 1.0,
        "FEAT_WINDOW_SEC": 60.0,
        "FEAT_STEP_SEC": 1.0,
        "LABEL_PURITY": 0.9,
        "KEEP_LABELS": [1, 2, 3],
        "CHEST_CHANNELS": ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"],
    },
    "LOG": {
        "CKPT_DIR": "checkpoints",
        "PLOT_DIR": "plots",
    },
    "CLASS_NAMES": ["Baseline", "Stress", "Amusement"],
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_config(yaml_path=None):
    cfg = copy.deepcopy(DEFAULT)
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)
        cfg = _deep_merge(cfg, user)
    return cfg
