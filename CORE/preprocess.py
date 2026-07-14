import argparse

from config import get_config
from dataset.preprocessing import WESADPreprocessor, extract_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--features", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.config)
    if args.raw_dir is not None:
        cfg["PREPROCESS"]["RAW_DIR"] = args.raw_dir

    do_raw = args.raw or not args.features
    do_feat = args.features

    if do_raw:
        print(f">>> [raw] WESAD {cfg['PREPROCESS']['RAW_DIR']} "
              f"({cfg['PREPROCESS']['ORIG_FS']} -> {cfg['PREPROCESS']['TARGET_FS']} Hz)")
        WESADPreprocessor(cfg).run()

    if do_feat:
        print(f">>> [features] WESAD {cfg['PREPROCESS']['RAW_DIR']} "
              f"(window {cfg['PREPROCESS'].get('FEAT_WINDOW_SEC', 60.0)}s)")
        extract_features(cfg)


if __name__ == "__main__":
    main()
