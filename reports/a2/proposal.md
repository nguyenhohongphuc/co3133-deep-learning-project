# A2 Dataset Proposal — Group NNs

CO3133 · Semester 261 · Track: **Image classification** (handbook §17)

Revision 2 · 28/10/2026. Revision 1 was submitted on 07/10/2026. This revision replaces the
split and leakage numbers with those from the final EDA (`notebooks/a2_eda.ipynb`).
Dataset, task, metrics and planned models are unchanged.

Every number below comes from `notebooks/a2_eda.ipynb` (run end-to-end on the local copy of the
dataset) or from the compute benchmark of 07/10/2026.

## 1. Dataset name, source, version, license

- **Name:** 30VNFoods (Vietnamese Foods)
- **Source:** Kaggle `quandang/vietnamese-foods` — <https://www.kaggle.com/datasets/quandang/vietnamese-foods>;
  paper: <https://ieeexplore.ieee.org/abstract/document/9530774>
- **Version:** Kaggle version **11** (pinned in the notebook: `quandang/vietnamese-foods/versions/11`).
  Archive SHA-256: `fa49ef9e176949df8d42b14ee1e870501f4dd31765be27bf15828591e1ff0a41`.
- **License:** MIT (Copyright 2021 Quan Dang). Images were crawled from the web; the source URLs are
  shipped in the `Urls/` folder, so copyright of individual images stays with their sources.
  We use the images for coursework only and do not redistribute them.

## 2. Task, input and output

- **Task:** single-label classification into 30 Vietnamese dishes.
- **Input:** one RGB image. Training: `RandomResizedCrop(224)`. Evaluation: `Resize(256)` + `CenterCrop(224)`.
- **Output:** 30 class logits; prediction = argmax.

## 3. Number of samples and annotation types

- **25,136 images, 30 classes.** One class label per image, given by its folder name. No other annotations.
- Official split: **Train 17,581 · Validate 2,515 · Test 5,040**, the same 30 classes in each.
- All 25,136 files decode; all are RGB.
- Meets §19.1: 30 ≥ 5 classes; 17,164 ≥ 5,000 training images after cleaning; the smallest class
  (Banh pia) still has 306 train / 44 validate / 89 test images.

## 4. Preliminary distribution analysis

| | Value |
|---|---|
| Images per class (official Train) | min 310 (Banh pia), max 1,071 (Bun bo Hue), mean 586 |
| Imbalance (max / min) | 3.45×; the official split is ≈ 70 / 10 / 20 inside every class |
| Image size (Train) | median 1000 × 720 px; min 100 × 83; max 7360 × 5184 |
| Aspect ratio (Train) | median 1.33; 24.6% exactly square; 1.1% extreme (> 2 or < 0.5) |
| Short side < 224 px | 1.4% (251 images), at most 2.8% in any class (Canh chua) |
| Colour (Train v2) | mean RGB [0.582, 0.514, 0.400], std [0.269, 0.265, 0.293]: warmer and brighter than ImageNet |
| Exact duplicates (MD5) | 63 groups / 129 images; 40 groups span two splits, 38 carry two different labels |
| Near duplicates (pHash) | 427 clusters of ≥ 2 images; 222 span two splits, 56 carry two different labels |

Observations that shape the plan:

- **Visually close class pairs.** Label conflicts among duplicates concentrate on Bun bo Hue / Bun rieu,
  Banh chung / Banh tet, Banh duc / Banh gio and Banh can / Banh khot. We expect these to be the main
  confusions.
- **Brightness does not separate classes.** Per-class median brightness lies in 0.41–0.57 and the
  boxes overlap. Banh pia is the brightest class: many of its photos show packaging on white
  backgrounds, which is a possible shortcut.

Figures: `results/a2/figures/a2_class_distribution.png`, `a2_aspect_ratio.png`, `a2_brightness_by_class.png`.

## 5. Data-splitting plan

- Keep the official Train / Validate / Test split, which is already stratified, then remove the images
  listed in Section 11.
- **Final split (v2): Train 17,164 · Validate 2,481 · Test 4,984 (24,629 images).** Class imbalance
  stays at 3.4×.
- Validate is used only for checkpoint selection and early stopping. Test is used once, for final
  evaluation. We also report Test metrics on the official (uncleaned) test set for comparison.
- Stored in `results/a2/splits/split_v2.json` (MD5 `8300fb207db7794f720e32e69283ab02`). It lists
  relative path, label and MD5 per image. Every model reads this same file. The split does not
  depend on a random seed; seeds are used only for training.

## 6. Split unit used to prevent leakage

**Near-duplicate cluster.** Every image gets a 64-bit perceptual hash (pHash). Two images are linked
when their Hamming distance is ≤ 6, or ≤ 8 when they carry the same label. Linked images form one
cluster, and a whole cluster is kept in exactly one split.

How we chose the threshold: we inspected sample pairs at distances 2, 4, 6 and 8 by eye. Up to 6,
cross-label pairs are rare (1–2 per distance after exact copies). At distance 8 they jump to 18,
mostly false matches, so distance 8 is accepted only within one class. The largest resulting
cluster has 3 images, so no chaining.

Without this step, 222 clusters would sit in more than one split.

## 7. Evaluation metrics

- **Main metrics (§20.1):** **Accuracy** and **macro-F1**. Macro-F1 is primary because of the 3.4× imbalance.
- Also reported: per-class F1 and the confusion matrix.
- Checkpoint = epoch with the best **validation macro-F1**.

## 8. Baseline plan

**CNN from Assignment 1, scaled up to 224 px RGB**, trained from scratch:

- Conv-BN-ReLU-MaxPool blocks with 32 → 64 → 128 → 256 channels.
- **Global average pooling** before the linear head. The A1 design (2 blocks + flatten) would put
  about 6 M parameters into the head at 224 px and see only about 10 px per output unit.
- About 0.4–1 M parameters.
- Normalised with Train v2 statistics (Section 4), same augmentation as the pretrained model.
- Adam, up to 30 epochs, early stopping on validation macro-F1.

## 9. Planned pretrained model(s)

**ResNet-50** pretrained on ImageNet (torchvision `IMAGENET1K_V2`). The 1000-way head is replaced by a
30-way linear head; 23.57 M parameters in total. ImageNet normalisation.

**Fine-tuning protocol:**

- **Frozen:** backbone frozen, only the head is trained (61 k parameters), lr 1e-3.
- **Full:** all layers trained; backbone lr 1e-4, head lr 1e-3.
- Shared settings: AdamW with cosine schedule; augmentation `RandomResizedCrop(224)` +
  `HorizontalFlip` + light `ColorJitter`; up to 15 epochs; best validation macro-F1.

**Controlled experiment (§20, §21):** frozen backbone vs. full fine-tuning. Only the set of trainable
layers changes, together with the learning rate that goes with it. Data, split, augmentation,
epochs, seeds and metrics stay fixed.

- Hypothesis: full fine-tuning gains macro-F1, because 30VNFoods differs from ImageNet in colour and
  content.
- Decision criterion: the difference counts only if it exceeds the spread across 3 seeds.

**Candidate extra ablation** (only if time allows): weighted vs. plain cross-entropy, to address the
3.4× imbalance.

## 10. Compute estimate

Measured on 07/10/2026: NVIDIA RTX 4060 Laptop GPU (8 GB), batch 64, 224 px, mixed precision,
PyTorch 2.14.1 / CUDA 12.6. Time per epoch is scaled to the 17,164 cleaned training images.

| Model | Throughput (img/s) | Time / epoch | Peak GPU memory | Epochs × seeds | GPU-hours |
|---|---|---|---|---|---|
| CNN baseline\* | 118 | 2.4 min | 3.0 GB | 30 × 3 | 3.6 |
| ResNet-50 frozen | 145 | 2.0 min | 0.7 GB | 15 × 3 | 1.5 |
| ResNet-50 full fine-tune | 58 | 4.9 min | 3.1 GB | 15 × 3 | 3.7 |
| Debugging margin (+30%) | | | | | 2.6 |
| **Total** | | | | | **≈ 11.4** |

\* The baseline was benchmarked with a 1.18 M-parameter 4-block CNN with global average pooling,
the same size class as the planned baseline. We will re-measure it once the final baseline is fixed.

- The optional weighted-CE ablation would add about 3.7 GPU-hours (one more full fine-tune × 3 seeds).
- Inference latency (batch 1): CNN 2.8 ms, ResNet-50 8.5–9.1 ms.
- Storage: dataset 4.3 GB; checkpoints about 0.6 GB (6 ResNet-50 runs × about 95 MB, plus the CNN).
- Fallback: Google Colab T4.

## 11. Subset selection rules

The full dataset is used. The only images removed are the 507 (2.0%) listed in
`results/a2/tables/a2_drop_list.csv`, which gives the reason and matching method for each:

| Rule | Exact (MD5) | Near (pHash) | Removed |
|---|---|---|---|
| Cluster carries two different labels → drop the whole cluster | 81 | 37 | 118 |
| Cluster spans splits → keep one copy, priority Test > Validate > Train | 14 | 182 | 196 |
| Cluster inside one split → keep one copy | 10 | 183 | 193 |
| **Total** | **105** | **402** | **507** |

Per split, this removes 417 images from Train, 34 from Validate and 56 from Test. The class that
loses the largest share is Banh chung (6.3%).

## 12. Known limitations

- **Source leakage not checked.** Images from the same website may still sit in different splits.
  The `Urls/` folder could map images to source domains, but we have not verified that mapping.
- **Label noise remains.** We only remove label conflicts that duplicates reveal. Other mislabelled
  images (for example, a goi cuon photo labelled Banh mi) may remain in the dataset.
