# Stress-UDA-Toolbox

Module hoá 6 script gốc (baseline + DANN + CDAN, trên tín hiệu thô WESAD và trên
đặc trưng .csv) thành một pipeline thống nhất theo phong cách **rPPG-Toolbox**:
`config (YAML) → data_loader → model → trainer → evaluation`.

## Cấu trúc

```
stress_uda_toolbox/
├── main.py                     # điểm vào: config → data → train → test → plot
├── config.py                   # default config + merge YAML
├── configs/                    # 6 config, mỗi cái = 1 script gốc
│   ├── baseline_cnn.yaml        (base1)   baseline_mlp.yaml (base2)
│   ├── dann_cnn.yaml (3)        dann_mlp.yaml (1)
│   └── cdan_cnn.yaml (8)        cdan_mlp.yaml (2)
├── preprocess.py               # WESAD .pkl gốc -> .npy (+ 2.csv nếu --features)
├── dataset/
│   ├── base_loader.py          # split subject, chuẩn hoá, class-weight, dataloader
│   ├── wesad_raw_loader.py     # .npy → CNN
│   ├── wesad_feature_loader.py # .csv → MLP
│   └── preprocessing/
│       ├── wesad_preprocess.py # đọc chest 8 kênh, resample 700→70Hz, windowing
│       └── feature_extract.py  # window 60s -> 62 đặc trưng sinh lý (neurokit2) -> 2.csv
├── neural_methods/
│   ├── model/  grl · backbones (CNN/MLP) · heads · discriminator
│   ├── loss/   dann · cdan  (self-contained, KHÔNG cần dalib)
│   └── trainer/ base_trainer · da_trainers (baseline/dann/cdan)
└── evaluation/ metrics · visualization (t-SNE, curve, confusion)
```

## Ánh xạ script gốc → config

| Script gốc | Modality | Method | Config |
|---|---|---|---|
| base1 | raw .npy → CNN | baseline | `baseline_cnn.yaml` |
| base2 | feature .csv → MLP | baseline | `baseline_mlp.yaml` |
| 3     | raw .npy → CNN | DANN | `dann_cnn.yaml` |
| 1     | feature .csv → MLP | DANN | `dann_mlp.yaml` |
| 8     | raw .npy → CNN | CDAN | `cdan_cnn.yaml` |
| 2     | feature .csv → MLP | CDAN | `cdan_mlp.yaml` |

## Bước 0 — Preprocessing (WESAD gốc → .npy / .csv)

Nếu chưa có sẵn `wesad_X_raw_70Hz.npy` và `2.csv`, sinh chúng từ dữ liệu WESAD
chính thức (các file `S2.pkl … S17.pkl`):

```bash
# đặt các file SX.pkl vào thư mục WESAD/ (hoặc trỏ --raw-dir tới nơi khác)

# nhánh CNN: sinh .npy (chest 8 kênh, 700->70Hz, window 2s)
python preprocess.py --config configs/preprocess.yaml --raw-dir WESAD --raw

# nhánh MLP: sinh 2.csv (62 đặc trưng sinh lý, window 60s, cần neurokit2) — chậm
python preprocess.py --config configs/preprocess.yaml --raw-dir WESAD --features
```

Nếu bạn đã có sẵn `2.csv` (như file bạn cung cấp) thì bỏ qua bước `--features`,
cứ đặt `2.csv` cạnh toolbox và chạy thẳng các config MLP.

- Đọc **chest (RespiBAN) 8 kênh**: ACC×3, ECG, EMG, EDA, Temp, Resp.
- Resample **700 → 70 Hz**, cắt window `WINDOW_SEC` giây (mặc định 2s = 140 mẫu),
  bước `STEP_SEC` (mặc định 1s = chồng lấp 50%).
- Chỉ giữ window có nhãn thuần trong `{1,2,3}` (độ thuần ≥ `LABEL_PURITY`).
- Ghi ra `wesad_X_raw_70Hz.npy` (M,8,140), `..._labels.npy`, `..._subjects.npy`;
  cờ `--features` sinh thêm `2.csv`.
- **Không** z-score ở bước này (để loader fit-trên-source, tránh rò rỉ target).

Giả định & lưu ý:
- Giả định định dạng WESAD chính thức (`pickle`, `encoding='latin1'`,
  `d['signal']['chest']`, `d['label']` ở 700Hz). Nếu nguồn khác, sửa
  `dataset/preprocessing/wesad_preprocess.py`.
- `feature_extract.py` sinh **đúng 62 cột** khớp `2.csv` (ACC time+freq, EMG
  time+freq, EDA + SCL/SCR, TEMP, RSP, ECG HR, HRV) bằng **neurokit2**. Nó là
  bản *tái lập gần đúng*, KHÔNG bit-exact: window mặc định 60s (HRV/LF-HF cần
  ≥ ~60s), bước 1s — nếu tham số gốc khác thì số sẽ lệch. Vài định nghĩa
  (RSP_Power, RSP_Stretch, EDA_SCR_Area, ACC_AbsInt) mình đặt theo cách hợp lý
  nhất; chỉnh trong file nếu bạn có định nghĩa gốc.
- Đặc trưng ECG/HRV cần R-peak thật; trên tín hiệu quá nhiễu neurokit trả NaN
  và các cột đó sẽ = 0 (đã bắt lỗi để không crash).
- **`2.csv` bạn gửi đã được standardize toàn cục** (mean≈0, std≈1) — tức là scale
  đã fit trên CẢ target trước khi split → rò rỉ nhẹ ở bước tạo file. Loader vẫn
  re-standardize fit-trên-source nên train được, nhưng con số tuyệt đối mang dấu
  vết đó. Bản extractor này xuất **giá trị thô chưa standardize** (đúng hơn về
  phương pháp luận); loader sẽ tự chuẩn hoá fit-trên-source.

## Chạy

```bash
pip install -r requirements.txt

python main.py --config configs/dann_cnn.yaml
python main.py --config configs/cdan_mlp.yaml --epochs 30 --device cpu
```

Kết quả: checkpoint trong `checkpoints/`, ba plot (history, confusion, t-SNE)
trong `plots/`.

## Thêm method / dataset mới

- Method mới: thêm 1 class kế thừa `BaseTrainer`, chỉ cài `_setup_da()` và
  `_train_step()`, đăng ký vào `neural_methods/trainer/__init__.py`.
- Dataset mới: thêm loader kế thừa `BaseLoader`, cài `_read_raw()` và
  `_normalize()`, đăng ký vào `dataset/__init__.py`.

---

## Những thay đổi có chủ ý so với code gốc (đọc kỹ trước khi so sánh số)

Trong lúc gộp, mình đã sửa vài chỗ khiến kết quả các script gốc **không so sánh
công bằng với nhau**. Nếu cần tái hiện đúng con số cũ, có thể chỉnh lại qua config.

1. **[ĐÍNH CHÍNH] Split của `base2`.** Trước đây mình nói base2 (`sid<=11` /
   `sid>=13`) bỏ sót subject 12 và tạo tập khác các script DA — **điều này SAI**.
   WESAD không có subject 12 (dataset chỉ gồm S2–S17 trừ S12), nên tập subject
   thực tế là {2..11, 13..17}; `sid<=11` = {2..11} (10 subj) và `sid>=13` =
   {13..17} (5 subj) **trùng khớp** với `sids[:10]` / `sids[10:]` của 1.py/2.py.
   Vậy base2 KHÔNG hề split khác. Toolbox vẫn thống nhất leave-subjects-out qua
   `DATA.NUM_SOURCE_SUBJECTS` (cùng cho ra 10 source / 5 target), nhưng đây là
   để nhất quán về cơ chế, không phải để sửa lỗi.

2. **Rò rỉ thống kê target khi chuẩn hoá** ở 3.py/8.py: chúng z-score bằng
   `mean/std` tính trên **toàn bộ** dữ liệu (gồm cả target). Toolbox chỉ fit
   thống kê trên **source** rồi transform target — đúng chuẩn UDA.

3. **Chọn model bằng cách nhìn trộm test.** 8.py (và base2) lưu checkpoint theo
   `best target acc` — mà target chính là tập test → điểm báo cáo bị lạc quan
   hoá. Mặc định toolbox để `MODEL_SELECTION: last`. Đặt `best_target` nếu muốn
   tái hiện, nhưng nên coi đó là *thượng giới* chứ không phải hiệu năng thật.
   *(WESAD không có target có nhãn để làm validation, nên đây là hạn chế cố hữu
   của thiết lập này — lý tưởng là tách một phần source làm val, hoặc dùng tiêu
   chí không nhãn.)*

4. **Nhân lambda hai lần** ở 8.py: lambda vừa nằm trong GRL vừa nhân lại
   `loss = cls + lambd*da` → thực chất là λ². Toolbox áp lambda **một lần** trong
   GRL (đúng như 3.py), `loss = cls + da`.

5. **Ngân sách train khác nhau** khiến baseline yếu giả tạo: base1 chạy 30 epoch
   / lr 1e-4, phần còn lại 60 / 1e-3. Đã thống nhất 60 / 1e-3 cho mọi config.

6. **`8.py` gọi `run_tsne` chưa định nghĩa** (chỉ có `tsne_cdan_cnn`) → sẽ
   `NameError` ở cuối train. Toolbox thay bằng `evaluation.visualization.plot_tsne`.

7. **Gỡ phụ thuộc `dalib`** (và hack `np.float = float`): DANN/CDAN được viết lại
   self-contained trong `neural_methods/loss/`, không cần cài thư viện ngoài.

### Vẫn nên lưu ý (chưa xử lý, vì phụ thuộc thiết kế thí nghiệm của bạn)

- **Chỉ một lần split (một seed).** 10/… subject là một hoán vị cố định; nên
  chạy nhiều fold/seed rồi báo cáo trung bình ± độ lệch, vì với ~14 subject
  phương sai giữa các split thường rất lớn.
- **Class imbalance:** dùng class-weight là hợp lý, nhưng accuracy vẫn dễ gây
  hiểu nhầm — nên báo cáo **macro-F1** (đã có sẵn trong `metrics.py`) làm chỉ số
  chính, nhất là khi lớp *Stress* thường bị bỏ sót (thấy rõ trong smoke test).
- **`window`/độ dài chuỗi** giả định cố định; nếu dữ liệu thật có độ dài thay đổi
  cần xử lý padding riêng.
