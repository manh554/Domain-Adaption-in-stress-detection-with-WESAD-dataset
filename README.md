# Domain-Adaption-in-stress-detection-with-WESAD-dataset
Unsupervised domain adaptation for stress detection on WESAD, refactored from
6 standalone scripts into one modular pipeline (rPPG-Toolbox style):
`config (YAML) → data_loader → model → trainer → evaluation`.
Two modalities × three methods:
	Baseline (source-only)	DANN	CDAN
Raw signal (.npy) → CNN	`baseline_cnn`	`dann_cnn`	`cdan_cnn`
Features (.csv) → MLP	`baseline_mlp`	`dann_mlp`	`cdan_mlp`
Each cell maps to a config in `configs/` and to one of the original scripts
(base1, base2, 3, 1, 8, 2).
Structure
```
stress_uda_toolbox/
├── main.py                 # config → data → train → test → plots
├── preprocess.py           # WESAD .pkl → .npy (+ 2.csv)
├── config.py               # default config + YAML merge
├── configs/                # 6 model configs + preprocess.yaml
├── dataset/
│   ├── base_loader.py      # subject split, normalization, class weights
│   ├── wesad_raw_loader.py     # .npy → CNN
│   ├── wesad_feature_loader.py # .csv → MLP
│   └── preprocessing/
│       ├── wesad_preprocess.py # chest 8 ch, 700→70Hz, windowing
│       └── feature_extract.py  # 62 physiological features (neurokit2)
├── neural_methods/
│   ├── model/              # grl, backbones (CNN/MLP), head, discriminator
│   ├── loss/               # dann, cdan (self-contained, no dalib)
│   └── trainer/            # base_trainer + baseline/dann/cdan
└── evaluation/             # metrics, visualization (t-SNE, curves, confusion)
```
Setup
```bash
pip install -r requirements.txt
```
Step 0 — Preprocessing (optional)
Only needed if you don't already have `wesad_X_raw_70Hz.npy` / `2.csv`.
Put the official WESAD `S*.pkl` files in a folder, then:
```bash
# CNN path: build .npy (chest 8 channels, 700→70Hz, 2s windows)
python preprocess.py --config configs/preprocess.yaml --raw-dir WESAD --raw

# MLP path: build 2.csv (62 features, 60s windows, needs neurokit2 — slow)
python preprocess.py --config configs/preprocess.yaml --raw-dir WESAD --features
```
If you already have the `.npy`/`.csv`, skip this and just place them next to the
toolbox (or set paths in `DATA`).
Run
```bash
python main.py --config configs/dann_cnn.yaml
python main.py --config configs/cdan_mlp.yaml --epochs 30 --device cpu
```
Outputs: checkpoint in `checkpoints/`, three plots (history, confusion, t-SNE)
in `plots/`.
Extending
New method: subclass `BaseTrainer`, implement `_setup_da()` and
`_train_step()`, register it in `neural_methods/trainer/__init__.py`.
New dataset: subclass `BaseLoader`, implement `_read_raw()` and
`_normalize()`, register it in `dataset/__init__.py`.
Notes
Normalization is fit on source only (raw path z-scored global stats in the
original 3.py/8.py, which leaked target statistics). Now source-only, then
applied to target — standard UDA.
Model selection defaults to `last` (`MODEL_SELECTION`). The original 8.py
saved the checkpoint with the best target accuracy, i.e. it peeked at the
test set. Set `best_target` to reproduce that, but treat the number as an
upper bound, not real performance.
Lambda is applied once (inside the GRL). 8.py multiplied by lambda a second
time (effectively λ²); fixed.
Self-contained DANN/CDAN — no `dalib` dependency, no `np.float` hack.
Feature extractor reproduces the exact 62 columns of `2.csv` via neurokit2,
but is an approximation, not bit-exact: window defaults to 60s, and a few
feature definitions (RSP_Power, RSP_Stretch, EDA_SCR_Area, ACC_AbsInt) are my
best guess. The provided `2.csv` is also already globally standardized (a mild
leak); the extractor outputs raw values and lets the loader standardize on
source.
Subject split: WESAD has no subject 12, so the different sid conditions in
the original scripts all resolve to the same 10-source / 5-target split. The
toolbox unifies this via `DATA.NUM_SOURCE_SUBJECTS`.
Still worth doing (not handled here): run multiple subject splits / seeds and
report mean ± std (variance is high with ~15 subjects), and prefer macro-F1
over accuracy given class imbalance.
