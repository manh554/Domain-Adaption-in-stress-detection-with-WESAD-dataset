import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_COLORS = ["#440154", "#21918c", "#fde725", "#3b528b", "#5ec962"]


def _savefig(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] saved -> {save_path}")
    plt.close(fig)


def plot_tsne(feats, labels, class_names, title="t-SNE",
              save_path=None, perplexity=30, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed)
    Z = tsne.fit_transform(feats)

    fig = plt.figure(figsize=(7, 6))
    for cls, name in enumerate(class_names):
        idx = labels == cls
        plt.scatter(Z[idx, 0], Z[idx, 1], s=8,
                    c=_COLORS[cls % len(_COLORS)], label=name, alpha=0.9)
    plt.title(title, fontsize=14)
    plt.xticks([]); plt.yticks([])
    plt.legend(frameon=False)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_history(history, title="Training curves", save_path=None):
    epochs = range(1, len(history["tgt_acc"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, history["cls"], label="cls loss", color="#3b528b")
    ax1.plot(epochs, history["da"], label="da loss", color="#5ec962")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["tgt_acc"], label="tgt acc", color="#440154", lw=2)
    ax2.set_ylabel("target acc (%)")
    ax2.legend(loc="upper right")

    plt.title(title)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_confusion(cm, class_names, title="Confusion matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.title(title)
    plt.tight_layout()
    _savefig(fig, save_path)
