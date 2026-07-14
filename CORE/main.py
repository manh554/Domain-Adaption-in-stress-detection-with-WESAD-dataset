import argparse
import os
import random

import numpy as np
import torch

from config import get_config
from dataset import build_loader
from neural_methods.trainer import build_trainer
from evaluation import confusion, plot_tsne, plot_history, plot_confusion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.config)
    if args.epochs is not None:
        cfg["TRAIN"]["EPOCHS"] = args.epochs

    set_seed(cfg["TRAIN"]["SEED"])
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f">>> Method={cfg['MODEL']['METHOD']} | Backbone={cfg['MODEL']['BACKBONE']} "
          f"| Loader={cfg['DATA']['LOADER']} | Device={device}")

    loader = build_loader(cfg)
    src_loader, tgt_loader, test_loader, meta = loader.build(device)
    print(f">>> source subjects={meta['src_subjects']} (n={meta['n_src']}) | "
          f"target subjects={meta['tgt_subjects']} (n={meta['n_tgt']})")

    trainer = build_trainer(cfg, meta, device)
    history = trainer.train(src_loader, tgt_loader, test_loader)

    class_names = cfg["CLASS_NAMES"]
    gts, preds = trainer.test(test_loader, target_names=class_names)

    if not args.no_plot:
        pdir = cfg["LOG"]["PLOT_DIR"]
        name = cfg["MODEL"]["NAME"]
        plot_history(history, title=f"{name} - training",
                     save_path=os.path.join(pdir, f"{name}_history.png"))
        plot_confusion(confusion(gts, preds), class_names,
                       title=f"{name} - confusion",
                       save_path=os.path.join(pdir, f"{name}_cm.png"))
        feats, labels = trainer.extract_features(test_loader)
        plot_tsne(feats, labels, class_names, title=f"{name} - t-SNE (target)",
                  save_path=os.path.join(pdir, f"{name}_tsne.png"))


if __name__ == "__main__":
    main()
