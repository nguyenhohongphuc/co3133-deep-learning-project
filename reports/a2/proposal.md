# A2 Dataset Proposal — Group NNs

CO3133 · Semester 261 · Track: **Image classification** (handbook §17)

Numbers below come from our runs of `scripts/a2_eda.py` (`results/a2/eda/`) and
`scripts/a2_benchmark.py` (`results/a2/compute/`) on 07/10/2026.

## 1. Dataset name, source, version, license

- **Name:** 30VNFoods (Vietnamese Foods)
- **Source:** Kaggle `quandang/vietnamese-foods` — <https://www.kaggle.com/datasets/quandang/vietnamese-foods>;
  paper: <https://ieeexplore.ieee.org/abstract/document/9530774>
- **Version:** Kaggle archive version 11, downloaded 07/10/2026; SHA-256 fingerprint
  `fa49ef9e176949df8d42b14ee1e870501f4dd31765be27bf15828591e1ff0a41`
- **License:** MIT

## 2. Task, input and output

- **Task:** single-label classification into 30 Vietnamese dishes.
- **Input:** one RGB image, resized/cropped to 3×224×224.
- **Output:** 30 class logits; prediction = argmax.

## 3. Number of samples and annotation types

- **25,136 images, 30 classes**, one class label per image (folder name). No corrupted files.
- Official split: **Train 17,581 · Validate 2,515 · Test 5,040**.
- Meets §19.1: ≥ 5 classes, ≥ 5,000 training images, smallest class still has 310 / 45 / 89 images.

## 4. Preliminary distribution analysis

| | Value |
|---|---|
| Images per class (train) | min 310 (Banh pia), max 1,071 (Bun bo Hue), mean 586 |
| Imbalance (max/min) | 3.45×, same in all three splits |
| Image size (median) | 1000 × 720 px; 1.38% have shorter side < 224 px |
| Duplicates (pHash distance ≤ 4) | 309 clusters; **159 span two splits**, **55 carry two different labels** |

Figures: `results/a2/eda/figures/` (class distribution, image sizes, samples, duplicate examples).

## 5. Data-splitting plan

- Keep the official Train / Validate / Test split (stratified, ≈ 70/10/20), then remove the
  duplicates listed in Section 11.
- Final split: **Train 17,275 · Validate 2,488 · Test 4,995**. Seed 42.
- Validate: checkpoint selection only. Test: final evaluation only.
  Saved in `results/a2/splits/split_seed42.json`; the same file is used by every model.

## 6. Split unit used to prevent leakage

**Near-duplicate cluster.** Images whose perceptual hash (pHash) differs by ≤ 4 bits are grouped;
a whole cluster must stay in one split. This matters here: 159 clusters cross the official splits
(107 Train–Test).

## 7. Evaluation metrics

**Accuracy** and **macro-F1** (§20.1), plus per-class F1 and confusion matrix.
Checkpoint = best validation macro-F1.

## 8. Baseline plan

Small CNN trained from scratch (4 conv blocks + global average pooling, 1.18 M parameters),
224 px input, 30 epochs.

## 9. Planned pretrained model(s)

**ResNet-50** pretrained on ImageNet (torchvision), new 30-class head, 23.57 M parameters.
Controlled experiment: **frozen backbone vs. full fine-tuning**.

## 10. Compute estimate

Measured on an NVIDIA RTX 4060 Laptop GPU (8 GB), batch 64, 224 px, mixed precision:

| Model | Time / epoch | Peak GPU memory | Epochs × seeds | GPU-hours |
|---|---|---|---|---|
| Small CNN (baseline) | 2.5 min | 3.0 GB | 30 × 3 | 3.7 |
| ResNet-50 frozen | 2.0 min | 0.7 GB | 15 × 3 | 1.5 |
| ResNet-50 full fine-tune | 5.0 min | 3.1 GB | 16 × 3 | 3.9 |
| Ablation (no augmentation) | 5.0 min | 3.1 GB | 15 × 3 | 3.8 |
| Debugging margin (+30%) | | | | 3.9 |
| **Total** | | | | **≈ 17** |

Storage: dataset 4.3 GB (1.1 GB after resizing to 256 px), checkpoints ≈ 1 GB.
Fallback: Google Colab T4.

## 11. Subset selection rules

Full dataset is used. Only these images are removed (list: `results/a2/eda/leakage_drop_list.csv`):

| Rule | Removed |
|---|---|
| Same image with two different labels → drop all copies | 116 |
| Same image in two splits → keep the Test/Validate copy, drop the others | 130 |
| Repeated copies inside one split → keep one | 132 |
| **Total** | **378 (1.5%)** |
