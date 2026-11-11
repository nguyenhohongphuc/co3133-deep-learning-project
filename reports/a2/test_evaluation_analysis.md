# A2 — Final test evaluation and error analysis

These results use fixed saved checkpoints on the cleaned test split: **4,984 images, 30 classes**. All seven checkpoints were evaluated. No training or hyperparameter selection used these test results.

## Protocol and run selection

Illustrative runs were fixed using the highest existing validation macro-F1 in each family, before test evaluation. All-seed means remain the main comparison; illustrative runs are not selected by test performance.

| Family | Illustrative run | Validation macro-F1 |
| --- | --- | --- |
| CNN baseline | a2_cnn_s42_1028-1546 | 0.5565 |
| ResNet-50 frozen | a2_resnet50_frozen_s44_1110-1151 | 0.7657 |
| ResNet-50 full | a2_resnet50_full_s44_1110-1204 | 0.8873 |

Inference: mps, PyTorch 2.8.0, batch 32, zero loader workers. Normalization and transforms come from each run config. CNN uses train statistics; ResNet uses ImageNet statistics. Each checkpoint is selected by validation macro-F1. Test predictions, checkpoint/config/split hashes and the device are saved per run.

## Quantitative comparison

| Family | Runs | Accuracy | Macro-F1 | Top-3 accuracy |
| --- | --- | --- | --- | --- |
| CNN baseline | 1 | 0.5451 | 0.5403 | 0.7542 |
| ResNet-50 frozen | 3 | 0.7679 ± 0.0013 | 0.7720 ± 0.0022 | 0.9169 |
| ResNet-50 full | 3 | 0.8908 ± 0.0064 | 0.8922 ± 0.0063 | 0.9683 |

± denotes sample standard deviation over seeds, not a confidence interval. CNN has one draft run only.

![a2_test_model_comparison.png](../../results/a2/figures/test/a2_test_model_comparison.png)

### Paired frozen/full comparison

| Seed | Frozen F1 | Full F1 | Full − frozen |
| --- | --- | --- | --- |
| 42 | 0.7698 | 0.8849 | +0.1150 |
| 43 | 0.7720 | 0.8957 | +0.1236 |
| 44 | 0.7742 | 0.8961 | +0.1219 |

Mean paired gain: **+0.1202 macro-F1**. Interpret consistency over these three seeds descriptively; no formal significance claim is made. This compares a frozen backbone (including fixed BatchNorm statistics) with full adaptation, not a learning-rate-only intervention.

CNN versus ResNet also changes architecture/capacity and initialization, so that comparison does not isolate pretraining alone.

## Per-class behavior and confusions

![a2_test_per_class_f1.png](../../results/a2/figures/test/a2_test_per_class_f1.png)

### CNN baseline

![a2_test_confusion_cnn.png](../../results/a2/figures/test/a2_test_confusion_cnn.png)

| True → predicted | Errors / true support | Rate within true class |
| --- | --- | --- |
| Banh canh → Hu tieu | 37 / 191 | 19.4% |
| Bun bo Hue → Bun rieu | 35 / 302 | 11.6% |
| Banh duc → Banh cuon | 31 / 129 | 24.0% |
| Bun bo Hue → Pho | 27 / 302 | 8.9% |
| Pho → Hu tieu | 27 / 162 | 16.7% |
| Cao lau → Bun dau mam tom | 26 / 122 | 21.3% |
| Bun thit nuong → Banh duc | 23 / 150 | 15.3% |
| Bun bo Hue → Banh mi | 22 / 302 | 7.3% |

### ResNet-50 frozen

![a2_test_confusion_resnet50_frozen.png](../../results/a2/figures/test/a2_test_confusion_resnet50_frozen.png)

| True → predicted | Errors / true support | Rate within true class |
| --- | --- | --- |
| Bun rieu → Bun bo Hue | 52 / 227 | 22.9% |
| Pho → Bun bo Hue | 32 / 162 | 19.8% |
| Banh canh → Bun bo Hue | 26 / 191 | 13.6% |
| Banh canh → Hu tieu | 20 / 191 | 10.5% |
| Hu tieu → Banh canh | 20 / 194 | 10.3% |
| Banh canh → Chao long | 17 / 191 | 8.9% |
| Bun mam → Bun bo Hue | 16 / 154 | 10.4% |
| Bun bo Hue → Bun rieu | 15 / 302 | 5.0% |

### ResNet-50 full

![a2_test_confusion_resnet50_full.png](../../results/a2/figures/test/a2_test_confusion_resnet50_full.png)

| True → predicted | Errors / true support | Rate within true class |
| --- | --- | --- |
| Banh canh → Hu tieu | 22 / 191 | 11.5% |
| Bun rieu → Bun bo Hue | 19 / 227 | 8.4% |
| Hu tieu → Banh canh | 13 / 194 | 6.7% |
| Bun bo Hue → Banh canh | 12 / 302 | 4.0% |
| Banh xeo → Banh khot | 11 / 234 | 4.7% |
| Hu tieu → Bun bo Hue | 11 / 194 | 5.7% |
| Banh chung → Banh tet | 10 / 99 | 10.1% |
| Banh duc → Banh cuon | 10 / 129 | 7.8% |

## Compute cost

| Family | Total / trainable parameters | Training minutes / run | Peak training GPU GiB | Forward ms / image |
| --- | --- | --- | --- | --- |
| CNN baseline | 397,086 / 397,086 | 20.85 | not recorded | 0.859 |
| ResNet-50 frozen | 23,569,502 / 61,470 | 12.13 | 0.4335751533508301 | 4.312 |
| ResNet-50 full | 23,569,502 / 23,569,502 | 24.97 | 2.8939061164855957 | 4.285 |

Training was measured on RTX 4060; current forward timing is on Apple MPS. Frozen and full use the same inference architecture and have similar measured forward costs. Small timing differences in a single pass do not establish a speed advantage. Training time includes validation; forward time excludes preprocessing and transfers.

## Paired illustrative predictions

| Outcome | Images | Fraction |
| --- | --- | --- |
| both_correct | 3704 | 74.3% |
| full_only_correct | 755 | 15.1% |
| frozen_only_correct | 130 | 2.6% |
| both_wrong | 395 | 7.9% |

These paired counts refer to the validation-selected illustrative frozen/full runs, not an ensemble. Both improvements and regressions appear in the galleries.

## Error groups

The following group analysis concerns **a2_resnet50_full_s44_1110-1204**, with **525 errors / 4984 test images**. Group-level seed variability is not estimated.

Groups below are overlapping diagnostics, not mutually exclusive semantic causes. Error rates use each group's own support, never raw error counts alone. Wilson intervals are descriptive binomial intervals within one selected run and do not represent seed variability.

| Group | Images | Errors | Error rate | 95% Wilson interval |
| --- | --- | --- | --- | --- |
| all | 4984 | 525 | 10.5% | [9.7%, 11.4%] |
| extreme_aspect | 52 | 4 | 7.7% | [3.0%, 18.2%] |
| normal_aspect | 4932 | 521 | 10.6% | [9.7%, 11.5%] |
| short_side_lt224 | 64 | 8 | 12.5% | [6.5%, 22.8%] |
| short_side_ge224 | 4920 | 517 | 10.5% | [9.7%, 11.4%] |
| dark_mean_lt025 | 35 | 3 | 8.6% | [3.0%, 22.4%] |
| bright_mean_gt075 | 48 | 4 | 8.3% | [3.3%, 19.6%] |
| confidence_ge09 | 4005 | 120 | 3.0% | [2.5%, 3.6%] |
| margin_le01 | 109 | 68 | 62.4% | [53.0%, 70.9%] |
| true_shape_pair | 234 | 26 | 11.1% | [7.7%, 15.8%] |
| true_soup_group | 720 | 118 | 16.4% | [13.9%, 19.3%] |
| extended_soup_group_exploratory | 1076 | 188 | 17.5% | [15.3%, 19.9%] |
| white_cake_family_exploratory | 624 | 85 | 13.6% | [11.2%, 16.5%] |
| pia_white_border_proxy | 11 | 0 | 0.0% | [0.0%, 25.9%] |
| pia_other_border | 78 | 2 | 2.6% | [0.7%, 8.9%] |

Operational thresholds: extreme aspect ratio >2 or <0.5; short side <224 px; brightness is mean RGB on a 64×64 thumbnail (<0.25 dark, >0.75 bright); confidence ≥0.9; top-1/top-2 probability margin ≤0.1. These thresholds describe test behavior and were not used to tune the classifier. The extended soup group adds Hu tieu and Pho to H2 after observing confusions; the exploratory white-cake family consists of Banh duc, Banh cuon, Banh gio and Banh bot loc. These are post-hoc descriptions.

### Error interpretation and possible improvements

| Diagnostic group | Measured frequency | Example IDs | Hypothesized cause | Future improvement idea |
| --- | --- | --- | --- | --- |
| Extended soup family | 188 errors / 1076 true images | 568, 2973, 4289 | Shared broth/toppings; discriminative noodle shape can be small or hidden | Review training labels and add varied, clearly identified examples of confusing dishes |
| Banh chung / Banh tet | 13 within-pair confusions / 234 true images | 650, 678 | Wrapped or close-up views can hide whole-item geometry; some labels need expert review | Audit ambiguous training labels and include whole-item plus cut-section views |
| White-cake family | 85 errors / 624 true images | 96, 978, 191 | Pale wrappers and minced toppings overlap; a few illustrated labels are ambiguous | Review labels without automatically relabeling test images; collect clearer shape cues in training |
| High-confidence predictions | 120 errors / 4005 predictions >=0.9 | 568, 96, 650 | Confidence can remain high for visually similar classes or possible annotation ambiguity | Assess calibration and any review threshold on validation, not on this final test |
| Extreme-aspect inputs | 4 errors / 52 images | 191, 3452, 442 | Center crop removes context in individual examples; no elevated group error rate is established | A future validation-controlled resize/letterbox comparison can test crop effects |
| Low-margin predictions | 68 errors / 109 margin <=0.1 cases | 4165, 4812, 3170 | Several competing labels receive similar scores; blur, occlusion or atypical framing may contribute | Inspect ambiguous training cases; evaluate uncertainty handling on validation |

High-confidence errors constitute 22.9% of this run's errors. The diagnostic groups overlap and these proposed causes are not causal findings.

## EDA hypotheses H1–H4

**H1 (Banh chung / Banh tet):** 13 within-pair confusions among 234 true images of these classes. See the directed confusion table for asymmetry.

**H2 (Bun bo Hue / Bun rieu / Banh canh):** 46 within-group confusions among 720 images. Broth color or hidden noodles are possible visual explanations, not labels inferred automatically from the confusion counts.

**H3 (extreme aspect ratio):** 4 errors among 52 extreme-aspect images. Using normal-aspect error rates within each true class gives 4.60 expected errors for 52 covered images. This adjustment does not establish that cropping caused the errors.

Observed extreme/normal error rates are 7.7% / 10.6%. For this illustrative full run, the original claim that extreme-aspect images fail more often is **not supported**; the extreme subset is small and its interval is wide.

![a2_test_crop_cases.png](../../results/a2/figures/test/a2_test_crop_cases.png)

**H4 (Banh pia packaging / white background):** the white-border proxy measures the fraction of near-white pixels in the outer 20% of a 64×64 image. Near-white means all RGB channels ≥220/255 and their range ≤30/255; a fraction ≥0.5 defines the proxy-positive group. Plates and food may also be white, so this is not a packaging annotation. Compare the two Banh pia groups above and inspect Grad-CAM examples. Without semantic region annotations or a controlled occlusion study, reliance on packaging/background remains unresolved.

## Qualitative evaluation

All galleries show the exact Resize/CenterCrop input, with predictions from the same three illustrative runs. Selection is deterministic: correct cases use margin ≥0.5 and path hashing with class diversity; hard-correct cases have the smallest top-1/top-2 margins; failure cases prioritize frequent confusion pairs and confidence. Improvements and regressions are both shown. Full manifests retain every selected image path and selection rule.

### Correct

![a2_test_examples_correct.png](../../results/a2/figures/test/a2_test_examples_correct.png)

### Hard Correct

![a2_test_examples_hard_correct.png](../../results/a2/figures/test/a2_test_examples_hard_correct.png)

### Wrong

![a2_test_examples_wrong.png](../../results/a2/figures/test/a2_test_examples_wrong.png)

### High Confidence Wrong

![a2_test_examples_high_confidence_wrong.png](../../results/a2/figures/test/a2_test_examples_high_confidence_wrong.png)

### Improvements

![a2_test_examples_improvements.png](../../results/a2/figures/test/a2_test_examples_improvements.png)

### Regressions

![a2_test_examples_regressions.png](../../results/a2/figures/test/a2_test_examples_regressions.png)

## Grad-CAM diagnostics

Maps use gradient-weighted final-convolution activations followed by ReLU ([Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391)). They are coarse explanatory maps, not segmentation masks or causal proof. Target classes are fixed from the saved predictions/ground truth. No weights or BatchNorm statistics are updated.

![a2_test_gradcam_1673.png](../../results/a2/figures/test/a2_test_gradcam_1673.png)

![a2_test_gradcam_1669.png](../../results/a2/figures/test/a2_test_gradcam_1669.png)

![a2_test_gradcam_1623.png](../../results/a2/figures/test/a2_test_gradcam_1623.png)

![a2_test_gradcam_1636.png](../../results/a2/figures/test/a2_test_gradcam_1636.png)

![a2_test_gradcam_0650.png](../../results/a2/figures/test/a2_test_gradcam_0650.png)

![a2_test_gradcam_1833.png](../../results/a2/figures/test/a2_test_gradcam_1833.png)

![a2_test_gradcam_2973.png](../../results/a2/figures/test/a2_test_gradcam_2973.png)

![a2_test_gradcam_2943.png](../../results/a2/figures/test/a2_test_gradcam_2943.png)

![a2_test_gradcam_3152.png](../../results/a2/figures/test/a2_test_gradcam_3152.png)

![a2_test_gradcam_0191.png](../../results/a2/figures/test/a2_test_gradcam_0191.png)

![a2_test_gradcam_0155.png](../../results/a2/figures/test/a2_test_gradcam_0155.png)

![a2_test_gradcam_0568.png](../../results/a2/figures/test/a2_test_gradcam_0568.png)

## Visual observations and improvement ideas

The observations below come from inspection of the exported galleries/Grad-CAM figures. They are illustrative rather than prevalence estimates; suspected annotation issues are not relabeled.

| Image / group | Observation | Hypothesized cause | Improvement idea |
| --- | --- | --- | --- |
| Test/Bun bo Hue/1057.jpg | Noodles, meat and broth are visible together; all three runs predict Bun bo Hue correctly. | Distinctive whole-dish cues are retained in this crop. | Retain varied whole-dish views in training. |
| Test/Banh mi/1329.jpg | The bread cross-section and filling are prominent. Both ResNets predict Banh mi; CNN predicts Bun thit nuong. | The baseline may emphasize ingredient texture over the bread silhouette. | Include varied bread shapes and ingredient contexts in training. |
| Test/Banh tet/242.jpg | Round cut sections, outer leaf and central filling are visible; all runs predict Banh tet. | Cross-section geometry provides a clear cue in this example. | Include whole and cut-section views when collecting training examples. |
| Test/Goi cuon/500.jpg | Only part of the roll is visible above a large dipping-sauce bowl; full is correct at 45.3% confidence. | Sauce and partial visibility create competing visual cues. | Inspect partial-roll examples and varied framing in training. |
| Test/Pho/523.jpg | Herbs cover much of the noodles; full predicts Pho at 31.1%, while frozen predicts Bun thit nuong. | Occlusion limits the visibility of noodle and broth cues. | Collect training examples with different garnish and occlusion levels. |
| Test/Bun thit nuong/419.jpg | The input is blurry and dominated by greens and shredded vegetables; full remains correct at 42.9%. | Image quality and ingredient overlap can make this an ambiguous low-margin case. | Evaluate blur robustness on validation and diversify training quality. |
| Test/Banh canh/799.jpg | White noodles, mixed sliced meats and egg share the crop; full and frozen predict Hu tieu although the label is Banh canh. | Shared toppings and an uncertain noodle-width cue can explain this confusion; the label is not changed. | Review ambiguous training soup labels with a knowledgeable annotator. |
| Test/Bun rieu/412.jpg | Reddish broth, pork and dark cubes are visible; all runs choose Bun bo Hue against the Bun rieu label. | The visible ingredients overlap between the two soup classes; annotation ambiguity is possible. | Review comparable training images and require explicit dish identification. |
| Test/Hu tieu/30.jpg | Meat and egg dominate the bowl, while only some noodles are visible; the ResNets predict Banh canh against Hu tieu. | Close framing can hide discriminative noodle geometry. | Include clear noodle views and broader framing in training. |
| Test/Banh beo/567.jpg | White filled rolls with minced topping are shown; all three runs predict Banh cuon against a Banh duc label. | The appearance suggests a case for label review rather than automatic attribution to model failure alone. | Flag for expert annotation review; do not automatically relabel test data. |
| Test/Banh chung/297.jpg | A tall leaf-wrapped cylindrical parcel is prominent; Grad-CAM overlaps the wrapped body for the Banh tet prediction. | Whole-item shape conflicts with the Banh chung label or with the learned class convention. | Review ambiguous wrapped training items and document class conventions. |
| Test/Banh chung/427.jpg | The crop shows a cut yellow filling and wrapper fragments rather than the whole parcel. Frozen is correct; full predicts Banh tet. | The global shape cue is missing and the two rice-cake interiors overlap. | Include both wrapped and cut-section training views. |
| Test/Banh duc/315.jpg | White folded cake pieces and minced topping occupy the center; full predicts Banh gio while frozen and CNN predict Banh duc. | Fine-tuning can regress on visually similar white-cake presentations. | Audit class boundaries and add diverse clearly labeled training presentations. |
| Test/Goi cuon/278.jpg | Pale translucent rolls fill the crop. Frozen and CNN predict Goi cuon correctly; full predicts Banh cuon. | Wrapper appearance and weak visible filling can overlap with the competing class. | Include variation in wrapper translucency and filling visibility in training. |
| Test/Banh bot loc/410.jpg | The original panoramic image includes condiments and the whole tray; center crop retains mostly a mixture of white cakes. | Context is removed in this individual crop, but the aggregate extreme-aspect group does not show higher error. | Test resize/letterbox alternatives on validation in a future controlled study. |
| Test/Canh chua/390.jpg | The original shows a broad fish-soup plate; the crop emphasizes herbs and only part of the fish and broth. | Reduced ingredient context is a plausible contributor to this individual Canh chua error. | Compare full-dish-preserving preprocessing on validation, without changing this final test result. |
| Test/Banh canh/210.jpg | Center crop enlarges bean sprouts and herbs and removes part of the surrounding bowl. | Shared soup toppings can dominate over noodle identity in this view. | Use validation to assess preprocessing that preserves the full dish. |
| Test/Banh cuon/1013.jpg | Small white rolls are heavily covered with herbs and toppings, with fruit at the edge; full correctly predicts Banh cuon. | Adapted features may help on an atypical mixed-context plate; one example does not establish the mechanism. | Maintain varied plate composition in training. |
| Test/Banh duc/296.jpg | A small takeaway serving is covered with minced meat and fried topping, alongside condiment containers; full predicts Banh duc correctly. | Dish-specific texture appears useful despite substantial contextual clutter. | Include realistic takeaway presentations in training. |
| Test/Banh pia/301.jpg | The pastries are small and out of focus in a tea setting; Grad-CAM for the wrong class includes cup and lower context. | Blur and a small discriminative food region plausibly contribute; this does not prove reliance on background. | Collect clearer and context-diverse Banh pia training views. |
| Test/Banh pia/340.jpg | Flat browned pastry surfaces dominate the crop; full predicts Banh xeo with only 37.8% confidence. | Local browned texture can overlap across pastry and pancake classes. | Inspect atypical Banh pia training appearances; preserve low-confidence cases for review. |
| Test/Banh pia/101.jpg | Correct Banh pia example: Grad-CAM overlaps the central stamped pastry and exposed filling. | The map is compatible with food-region evidence and does not demonstrate a white-background shortcut. | Use annotated food/background regions in any future causal shortcut study. |
| Test/Banh pia/146.jpg | Correct Banh pia example has a watermark over the pastries; Grad-CAM spreads across the central food region. | Food, stamp and text overlap spatially, so their separate influence cannot be inferred from this map. | Use controlled watermark/background interventions on validation for future diagnosis. |

## Limitations and conclusion

Results cover the cleaned test split and these fixed checkpoints. The CNN baseline is a single draft run. Error groups overlap; sparse groups have uncertain rates. Source-domain leakage and remaining label noise were not resolved by this analysis. The original uncleaned test set is not evaluated here. No test-driven model changes are made.

Training costs come from RTX 4060 logs; current inference costs come from a different device and cannot be treated as measurements on one shared device. Forward timings use one warmup batch and a single streamed pass; they exclude preprocessing/transfers and are not repeated latency benchmarks.

Frozen/full mean test macro-F1: **0.7720 / 0.8922**. Use the paired results, error examples and compute costs together when drawing the final conclusion.

## Reproduce

```bash
python scripts/a2_evaluate_all.py --split test --device auto --batch-size 32
python scripts/a2_error_analysis.py
python scripts/a2_gradcam.py
python scripts/a2_error_analysis.py
```

The batch evaluator reuses complete matching exports by default. The second analysis pass incorporates the Grad-CAM manifest and visual review CSV without rerunning test inference. Required packages are already listed in requirements.txt (torch, torchvision, numpy, Pillow through torchvision, matplotlib, PyYAML).
